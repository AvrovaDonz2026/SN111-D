"""Deterministic offline benchmark for Claims ``agent_v1`` artifacts.

This module deliberately measures artifact properties only.  It does not call
miners, validators, model providers, Bittensor, or external URLs, and its
metrics must not be interpreted as network rewards.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

from pydantic import ValidationError


def _load_agent_v1_types() -> tuple[Any, Any, Any]:
    """Load the two pure artifact modules without importing miner/__init__.py.

    The repository's top-level miner package also imports optional runtime
    clients. Offline replay only needs the artifact Pydantic models and their
    deterministic validator, so an in-memory package stub keeps this utility
    independent of those optional dependencies.
    """

    try:
        from miner.agent_v1.artifact_models import Artifact, unknown_field_diagnostics
        from miner.agent_v1.artifact_validator import validate_agent_artifact

        return Artifact, validate_agent_artifact, unknown_field_diagnostics
    except Exception:
        for module_name in ("miner.agent_v1.artifact_validator", "miner.agent_v1.artifact_models", "miner.agent_v1", "miner"):
            sys.modules.pop(module_name, None)
        repo_root = Path(__file__).resolve().parents[2]
        miner_package = types.ModuleType("miner")
        miner_package.__path__ = [str(repo_root / "miner")]
        agent_package = types.ModuleType("miner.agent_v1")
        agent_package.__path__ = [str(repo_root / "miner" / "agent_v1")]
        sys.modules["miner"] = miner_package
        sys.modules["miner.agent_v1"] = agent_package
        from miner.agent_v1.artifact_models import Artifact, unknown_field_diagnostics
        from miner.agent_v1.artifact_validator import validate_agent_artifact

        return Artifact, validate_agent_artifact, unknown_field_diagnostics


Artifact, validate_agent_artifact, unknown_field_diagnostics = _load_agent_v1_types()


def _normalized_statement(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _validation_error_details(exc: ValidationError) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for error in exc.errors():
        details.append(
            {
                "path": ".".join(str(part) for part in error.get("loc", ())),
                "type": str(error.get("type", "validation_error")),
                "message": str(error.get("msg", "validation failed")),
            }
        )
    return details


def _runtime_metric(metadata: dict[str, Any], key: str) -> Any:
    runtime = metadata.get("runtime_metrics")
    if isinstance(runtime, dict) and runtime.get(key) is not None:
        return runtime[key]
    return metadata.get(key)


def _timing_and_cost(metadata: dict[str, Any]) -> dict[str, Any]:
    elapsed = _runtime_metric(metadata, "elapsed_seconds")
    cost = _runtime_metric(metadata, "cost_usd")
    cost_kind = _runtime_metric(metadata, "cost_kind") or metadata.get("usage_source")
    try:
        elapsed = float(elapsed) if elapsed is not None else None
    except (TypeError, ValueError):
        elapsed = None
    try:
        cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost = None
    return {
        "latency_seconds": elapsed,
        "latency_status": "measured" if elapsed is not None else "unavailable",
        "cost_usd": cost,
        "cost_kind": str(cost_kind) if cost is not None and cost_kind else ("unknown" if cost is not None else "unavailable"),
        "cost_status": "measured" if cost is not None else "unavailable",
    }


def _duplicate_metrics(artifact: Artifact) -> tuple[float, list[list[str]]]:
    groups: dict[str, list[str]] = {}
    for claim in artifact.logic.claims:
        key = _normalized_statement(claim.statement)
        if key:
            groups.setdefault(key, []).append(claim.claim_id)
    duplicate_groups = [ids for ids in groups.values() if len(ids) > 1]
    duplicate_rows = sum(len(ids) - 1 for ids in duplicate_groups)
    total = len(artifact.logic.claims)
    return (duplicate_rows / total if total else 0.0), duplicate_groups


def _claim_grounded(claim: Any, evidence_by_id: dict[str, Any]) -> bool:
    if claim.sources:
        return True
    return any(evidence_by_id.get(evidence_id) and evidence_by_id[evidence_id].source_refs for evidence_id in claim.evidence_ids)


def _evidence_complete(claim: Any, evidence_by_id: dict[str, Any]) -> bool:
    if not claim.evidence_ids:
        return False
    return all(
        evidence_by_id.get(evidence_id) is not None
        and claim.claim_id in evidence_by_id[evidence_id].linked_claim_ids
        for evidence_id in claim.evidence_ids
    )


def load_artifact_metrics(path: Path, *, expected_claim_ids: set[str] | None = None) -> dict[str, Any]:
    """Load one artifact and return JSON-serializable offline metrics."""

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "schema_valid": False,
            "schema_issue_count": 1,
            "schema_issues": [{"path": "", "type": "invalid_json", "message": str(exc)}],
            "validation_issue_count": 0,
            "validation_issues": [],
            "claims_total": None,
            "claim_coverage": None,
            "grounded_claims": None,
            "source_grounding_rate": None,
            "evidence_complete_claims": None,
            "evidence_link_completeness": None,
            "duplicate_rate": None,
            "duplicate_groups": [],
            **_timing_and_cost({}),
        }

    try:
        artifact = Artifact.model_validate(raw)
    except ValidationError as exc:
        details = _validation_error_details(exc)
        return {
            "schema_valid": False,
            "schema_issue_count": len(details),
            "schema_issues": details,
            "validation_issue_count": 0,
            "validation_issues": [],
            "claims_total": None,
            "claim_coverage": None,
            "grounded_claims": None,
            "source_grounding_rate": None,
            "evidence_complete_claims": None,
            "evidence_link_completeness": None,
            "duplicate_rate": None,
            "duplicate_groups": [],
            **_timing_and_cost(raw.get("metadata", {}) if isinstance(raw, dict) else {}),
        }
    except Exception as exc:
        return {
            "schema_valid": False,
            "schema_issue_count": 1,
            "schema_issues": [{"path": "", "type": "schema_validation_error", "message": str(exc)}],
            "validation_issue_count": 0,
            "validation_issues": [],
            "claims_total": None,
            "claim_coverage": None,
            "grounded_claims": None,
            "source_grounding_rate": None,
            "evidence_complete_claims": None,
            "evidence_link_completeness": None,
            "duplicate_rate": None,
            "duplicate_groups": [],
            **_timing_and_cost(raw.get("metadata", {}) if isinstance(raw, dict) else {}),
        }

    validation_issues = validate_agent_artifact(artifact)
    evidence_by_id = {record.evidence_id: record for record in artifact.evidence.records}
    claims = artifact.logic.claims
    grounded = sum(_claim_grounded(claim, evidence_by_id) for claim in claims)
    evidence_complete = sum(_evidence_complete(claim, evidence_by_id) for claim in claims)
    duplicate_rate, duplicate_groups = _duplicate_metrics(artifact)
    claim_ids = {claim.claim_id for claim in claims}
    coverage = None
    if expected_claim_ids is not None:
        coverage = len(expected_claim_ids & claim_ids) / len(expected_claim_ids) if expected_claim_ids else 1.0
    unknown_fields = unknown_field_diagnostics(raw)
    schema_issues = [
        {
            "path": str(issue.get("path", "")),
            "type": str(issue.get("code", "unknown_field")),
            "message": str(issue.get("message", "Unknown field")),
        }
        for issue in unknown_fields
    ]
    return {
        "schema_valid": not schema_issues,
        "schema_issue_count": len(schema_issues),
        "schema_issues": schema_issues,
        "validation_issue_count": len(validation_issues),
        "validation_issues": validation_issues,
        "validation_issue_codes": sorted({str(issue.get("code", "")) for issue in validation_issues}),
        "claims_total": len(claims),
        "claim_ids": sorted(claim_ids),
        "claim_coverage": coverage,
        "grounded_claims": grounded,
        "source_grounding_rate": grounded / len(claims) if claims else 0.0,
        "evidence_complete_claims": evidence_complete,
        "evidence_link_completeness": evidence_complete / len(claims) if claims else 0.0,
        "duplicate_rate": duplicate_rate,
        "duplicate_groups": duplicate_groups,
        **_timing_and_cost(artifact.metadata),
    }


def _delta(before: Any, after: Any) -> Any:
    if before is None or after is None:
        return None
    return round(float(after) - float(before), 8)


def benchmark_artifacts(paths: dict[str, Path]) -> dict[str, Any]:
    """Benchmark baseline/candidate/repaired files and return stable JSON data."""

    baseline_path = paths.get("baseline")
    baseline = load_artifact_metrics(baseline_path) if baseline_path else None
    if baseline and baseline.get("schema_valid"):
        # Baseline defines the reference claim set, so its own recall is 1.0.
        baseline["claim_coverage"] = 1.0
    expected_ids = set(baseline.get("claim_ids", [])) if baseline and baseline.get("schema_valid") else None
    artifacts: list[dict[str, Any]] = []
    for label, path in paths.items():
        metrics = baseline if label == "baseline" and baseline is not None else load_artifact_metrics(path, expected_claim_ids=expected_ids)
        artifacts.append({"label": label, "path": str(path), "metrics": metrics})

    comparisons: list[dict[str, Any]] = []
    if baseline is not None:
        for row in artifacts:
            if row["label"] == "baseline":
                continue
            metrics = row["metrics"]
            comparisons.append(
                {
                    "from": "baseline",
                    "to": row["label"],
                    "schema_gate": "passed" if baseline.get("schema_valid") and metrics.get("schema_valid") else "blocked",
                    "deltas": {
                        key: _delta(baseline.get(key), metrics.get(key))
                        for key in ("claim_coverage", "source_grounding_rate", "evidence_link_completeness", "duplicate_rate", "latency_seconds", "cost_usd")
                    },
                    "validation_issue_delta": (
                        (metrics.get("validation_issue_count") or 0) - (baseline.get("validation_issue_count") or 0)
                        if baseline.get("schema_valid") and metrics.get("schema_valid")
                        else None
                    ),
                }
            )
    return {
        "benchmark": "sn111-agent-v1-offline-replay",
        "version": 1,
        "deterministic": True,
        "artifacts": artifacts,
        "comparisons": comparisons,
        "limitations": [
            "Claim coverage uses exact baseline claim_id matches; it is not semantic recall.",
            "Grounding checks artifact source/evidence references only and does not fetch sources or judge quote meaning.",
            "Latency and cost are reported only when supplied by artifact metadata; unavailable values are not imputed.",
            "Metrics are offline diagnostics and are not miner rewards or online validator scores.",
        ],
    }


def write_csv(result: dict[str, Any], path: Path) -> None:
    fields = [
        "label", "path", "schema_valid", "schema_issue_count", "validation_issue_count", "claims_total",
        "claim_coverage", "source_grounding_rate", "evidence_link_completeness", "duplicate_rate",
        "latency_seconds", "latency_status", "cost_usd", "cost_kind", "cost_status", "validation_issue_codes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["artifacts"]:
            metrics = row["metrics"]
            values = {
                field: (";".join(metrics.get(field, [])) if field == "validation_issue_codes" else metrics.get(field))
                for field in fields
                if field not in {"label", "path"}
            }
            writer.writerow({"label": row["label"], "path": row["path"], **values})


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic offline agent_v1 artifact replay metrics.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--repaired", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--csv-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths = {"baseline": args.baseline, "candidate": args.candidate}
    if args.repaired:
        paths["repaired"] = args.repaired
    result = benchmark_artifacts(paths)
    encoded = json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True)
    if args.json_out:
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    else:
        print(encoded)
    if args.csv_out:
        write_csv(result, args.csv_out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
