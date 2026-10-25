"""Offline SN111-D scoring simulation.

This module intentionally does not import neurons or connect to Bittensor.  It
models the data gates that a validator-like process can apply to synthetic
paper/task responses and compares two deterministic selection strategies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Paper:
    paper_id: str
    title: str
    spans: dict[str, str]
    task_claim_ids: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    miner_id: str
    claim_ids: tuple[str, ...]
    source_span_ids: tuple[str, ...]
    provenance: str
    text: str
    quality: float


@dataclass(frozen=True)
class GateResult:
    quality: float
    provenance: float
    duplicate: float
    coverage: float
    grounding: float
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class CandidateScore:
    miner_id: str
    score_total: float
    accepted: bool
    score_components: dict[str, float]
    gates: GateResult


def synthetic_paper() -> Paper:
    return Paper(
        paper_id="paper-synthetic-001",
        title="Synthetic intervention study",
        spans={
            "s1": "The intervention increased recovery from 0.40 to 0.60.",
            "s2": "The estimate was based on 120 participants.",
            "s3": "The result is observational and does not establish causation.",
        },
        task_claim_ids=("c1", "c2", "c3"),
    )


def synthetic_candidates() -> list[Candidate]:
    return [
        Candidate("miner-good", ("c1", "c2", "c3"), ("s1", "s2", "s3"), "doi:10.1234/synthetic-001", "recovery +0.20; n=120; observational", 0.96),
        Candidate("miner-duplicate", ("c1", "c2", "c3"), ("s1", "s2", "s3"), "doi:10.1234/synthetic-001", "recovery +0.20; n=120; observational", 0.95),
        Candidate("miner-partial", ("c1",), ("s1",), "doi:10.1234/synthetic-001", "recovery +0.20", 0.86),
        Candidate("miner-ungrounded", ("c1", "c2"), ("missing-span",), "doi:10.1234/synthetic-001", "recovery +0.20; n=120", 0.90),
        Candidate("miner-untrusted", ("c1", "c2", "c3"), ("s1", "s2", "s3"), "unknown", "recovery +0.20; n=120; causal", 0.72),
    ]


def _fingerprint(candidate: Candidate) -> str:
    payload = "|".join((*candidate.claim_ids, *candidate.source_span_ids, candidate.text.strip().lower()))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_candidate(paper: Paper, candidate: Candidate, duplicate_fingerprints: set[str]) -> CandidateScore:
    reasons: list[str] = []
    quality = max(0.0, min(1.0, candidate.quality))
    provenance = 1.0 if candidate.provenance.startswith("doi:") else 0.0
    if provenance == 0.0:
        reasons.append("provenance_gate")
    fingerprint = _fingerprint(candidate)
    duplicate = 0.0 if fingerprint in duplicate_fingerprints else 1.0
    if duplicate == 0.0:
        reasons.append("duplicate_gate")
    coverage = len(set(candidate.claim_ids) & set(paper.task_claim_ids)) / len(paper.task_claim_ids)
    if coverage < 2 / 3:
        reasons.append("coverage_gate")
    grounding = sum(span in paper.spans for span in candidate.source_span_ids) / max(1, len(candidate.source_span_ids))
    if grounding < 1.0:
        reasons.append("grounding_gate")
    accepted = not reasons
    components = {
        "schema_validity": 1.0 if candidate.claim_ids and candidate.source_span_ids and candidate.text.strip() else 0.0,
        "structure": coverage,
        "grounding": grounding,
        "relations": 1.0 if coverage >= 2 / 3 else coverage,
        "ontology": provenance,
    }
    # A deterministic validator-like weighted score. Gates cap the result.
    raw = (quality * 0.30 + provenance * 0.15 + duplicate * 0.15 + coverage * 0.20 + grounding * 0.20)
    if not accepted:
        raw = min(raw, 0.49)
    return CandidateScore(candidate.miner_id, round(raw, 4), accepted, components, GateResult(quality, provenance, duplicate, round(coverage, 4), grounding, accepted, tuple(reasons)))


def compare_strategies(paper: Paper, candidates: list[Candidate]) -> dict[str, Any]:
    """Compare strict gate-first selection with quality-first selection."""
    scores: list[CandidateScore] = []
    seen: set[str] = set()
    for candidate in candidates:
        score = evaluate_candidate(paper, candidate, seen)
        # Duplicate fingerprints are tracked after the first submission, even
        # when the first submission later fails a different gate.
        seen.add(_fingerprint(candidate))
        scores.append(score)
    by_id = {score.miner_id: score for score in scores}
    strict = sorted((s for s in scores if s.accepted), key=lambda s: s.score_total, reverse=True)
    quality_first = sorted(scores, key=lambda s: s.gates.quality, reverse=True)
    return {
        "paper": asdict(paper),
        "candidates": [asdict(s) for s in scores],
        "strategies": {
            "strict_gate_first": {"selected": [s.miner_id for s in strict[:2]], "accepted_count": len(strict)},
            "quality_first": {
                "selected": [s.miner_id for s in quality_first[:2]],
                "accepted_selected": sum(s.accepted for s in quality_first[:2]),
            },
        },
        "notes": [
            "Offline synthetic simulation only; no wallet, network, mainnet, or TAO payout is involved.",
            f"Evaluated {len(by_id)} miner candidates against {len(paper.task_claim_ids)} task claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline SN111-D validator-like simulation")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    result = compare_strategies(synthetic_paper(), synthetic_candidates())
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("SN111-D offline simulation (synthetic data; no chain or wallet)")
    for strategy, details in result["strategies"].items():
        print(f"{strategy}: {', '.join(details['selected'])}")
    for item in result["candidates"]:
        gate_text = "pass" if item["gates"]["accepted"] else ",".join(item["gates"]["reasons"])
        print(f"  {item['miner_id']}: score={item['score_total']:.4f} gates={gate_text}")


if __name__ == "__main__":
    main()
