"""Run matched, model-backed verbose/compact eligibility replays; never write weights or backend data.

Run from the repository root with PYTHONPATH=. python scripts/compare_eligibility_contracts.py --help.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv

from validator.agent_v1.adjudication_models import AdjudicationContextBundle
from validator.agent_v1.comparison_models import BronzeDiffCase
from validator.agent_v1.file_agent_workflow import (
    FileAgentWorkflowConfig,
    FileAgentWorkflowSession,
)
from validator.agent_v1.model_usage import ModelUsageCollector
from validator.agent_v1.record_projection import project_agent_artifact


def read(path):
    return json.loads(Path(path).read_text())


def summarize(events):
    groups = defaultdict(
        lambda: {
            "events": 0,
            "known_output_events": 0,
            "failed_events": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
        }
    )
    for event in events:
        stage = event["stage_key"]
        component = next(
            (
                name
                for name in ("blind_reconstruction", "tiebreak", "negative", "positive")
                if name in stage
            ),
            stage,
        )
        row = groups[component]
        row["events"] += 1
        row["known_output_events"] += event.get("completion_tokens") is not None
        row["failed_events"] += event["status"] == "failed"
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens"):
            row[key] += event.get(key) or 0
    return dict(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--bronze-artifact", type=Path, required=True)
    parser.add_argument("--source-payload", type=Path, required=True)
    parser.add_argument("--miner-artifact", type=Path, required=True)
    parser.add_argument("--miner-id", default="uid_2")
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--case-limit",
        type=int,
        default=0,
        help="Zero replays the entire saved case set.",
    )
    parser.add_argument(
        "--contracts",
        nargs="+",
        choices=["verbose", "compact"],
        default=["verbose", "compact"],
    )
    args = parser.parse_args()
    if (args.output_dir / "comparison.json").exists():
        parser.error("Use a new output directory; comparison.json already exists.")
    load_dotenv(args.env_file, override=True)
    source = read(args.source_payload)
    bronze = read(args.bronze_artifact)
    miner = read(args.miner_artifact)
    case_data = read(args.case_file)
    candidates = [
        *project_agent_artifact(bronze, origin="bronze"),
        *project_agent_artifact(miner, origin="miner", miner_id=args.miner_id),
    ]
    by_id = {c.candidate_id: c for c in candidates}
    cases = [BronzeDiffCase.model_validate(c) for c in case_data["diff_cases"]]
    if args.case_limit:
        cases = cases[: args.case_limit]
    if not cases:
        parser.error("Fixture has no eligibility cases.")
    missing = {ref for case in cases for ref in case.candidate_ids if ref not in by_id}
    if missing:
        parser.error(f"Candidate identities cannot be recovered: {sorted(missing)}")
    paper_id = cases[0].paper_id
    if any(case.paper_id != paper_id for case in cases):
        parser.error("Use a single paper fixture.")
    source_map = {
        s["span_id"]: s["text"]
        for s in source["spans"]
        if s.get("span_id") and s.get("text")
    }
    contexts = [
        AdjudicationContextBundle(
            case=case,
            candidates=[by_id[c] for c in case.candidate_ids],
            source_context="\n".join(
                source_map.get(s, "")
                for c in case.candidate_ids
                for s in by_id[c].source_span_ids
            ),
        )
        for case in cases
    ]
    fixture = {
        "paper_id": paper_id,
        "contexts": [c.model_dump(mode="json") for c in contexts],
        "source_spans": source_map,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fixture, sort_keys=True).encode()
    ).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fixture.json").write_text(json.dumps(fixture, indent=2))
    config = FileAgentWorkflowConfig.from_env()
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "paper_id": paper_id,
        "case_count": len(cases),
        "fixture_sha256": fingerprint,
        "harness": config.adjudication_harness,
        "model_a": config.adjudication_negative_model,
        "model_b": config.adjudication_positive_model,
        "model_tiebreak": config.adjudication_tiebreak_model,
        "batch_size": config.adjudication_batch_size,
        "modes": {},
    }
    for mode in args.contracts:
        mode_dir = args.output_dir / mode
        collector = ModelUsageCollector(
            network="testnet",
            run_id=f"eligibility_compare_{mode}",
            batch_id=fingerprint[:16],
        )
        effective = replace(
            config,
            root=mode_dir,
            eligibility_output_contract=mode,
            eligibility_sample_verbose_paper=False,
            resume_existing_stages=False,
        )
        session = FileAgentWorkflowSession(
            config=effective,
            paper_id=paper_id,
            workspace_id="matched",
            candidates=candidates,
            paper_context=source.get("paper") or bronze.get("paper") or {},
            source_context_by_span_id=source_map,
            eligibility_source_context_by_span_id=source_map,
            usage_sink=collector.record,
        )
        print(
            f"Starting {mode}: {len(cases)} cases, fixture {fingerprint[:12]}",
            flush=True,
        )
        try:
            decisions = session.run_eligibility_adjudication(contexts)
            result = {
                "status": "complete",
                "decisions": [d.model_dump(mode="json") for d in decisions],
            }
        except Exception as exc:
            result = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "decisions": [],
            }
        events = collector.snapshot()
        mode_dir.mkdir(parents=True, exist_ok=True)
        (mode_dir / "usage_events.json").write_text(json.dumps(events, indent=2))
        (mode_dir / "decisions.json").write_text(json.dumps(result, indent=2))
        result["by_component"] = summarize(events)
        result["output_tokens"] = sum(e.get("completion_tokens") or 0 for e in events)
        report["modes"][mode] = result
        (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2))
        print(
            f"Finished {mode}: {result['status']}, output_tokens={result['output_tokens']}",
            flush=True,
        )
    if all(mode in report["modes"] for mode in ["verbose", "compact"]):
        baseline = report["modes"]["verbose"]["output_tokens"]
        report["output_reduction_pct"] = (
            round(100 * (1 - report["modes"]["compact"]["output_tokens"] / baseline), 2)
            if baseline
            else None
        )
        left = {d["case_id"]: d for d in report["modes"]["verbose"]["decisions"]}
        right = {d["case_id"]: d for d in report["modes"]["compact"]["decisions"]}
        report["selection_differences"] = [
            ref
            for ref in left.keys() & right.keys()
            if (left[ref]["selected_candidate_id"], left[ref]["resolution_status"])
            != (right[ref]["selected_candidate_id"], right[ref]["resolution_status"])
        ]
    (args.output_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    return 0 if all(r["status"] == "complete" for r in report["modes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
