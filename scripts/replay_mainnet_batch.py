"""Replay frozen mainnet Silver inputs without wallet, backend, or chain access."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import traceback

from dotenv import load_dotenv
from scripts.benchmark_equivalence_routing import (
    disable_response_cache,
    read,
    write,
    usage_summary,
    reliability,
    retained,
)
from validator.agent_v1.file_agent_workflow import (
    FileAgentWorkflowConfig,
    FileAgentSilverWorkflow,
)
from validator.agent_v1.model_usage import ModelUsageCollector
from validator.agent_v1.orchestrator import (
    MinerArtifactSubmission,
    run_paper_silver_pipeline,
)
from validator.agent_v1.record_projection import project_agent_artifact


def source_map(payloads):
    """Match validator source precedence and page-span aliases."""
    result = {}
    for payload in payloads:
        for i, row in enumerate(payload.get("spans", []), 1):
            sid = str(row.get("span_id") or row.get("id") or f"span_{i}").strip()
            text = str(row.get("text") or row.get("quote") or "").strip()
            if not sid or not text:
                continue
            result[sid] = text
            match = re.match(r"^(?P<prefix>.+-p\d{3})-(?P<suffix>markdown|001)$", sid)
            if match:
                alias = "001" if match["suffix"] == "markdown" else "markdown"
                result.setdefault(f"{match['prefix']}-{alias}", text)
    return result


def inputs(item, root):
    folder = root / item["fixture_dir"]
    bronze, source = read(folder / "bronze.json"), read(folder / "source.json")
    miners = [
        MinerArtifactSubmission(
            miner_id=mid, artifact=read(folder / f"miner_{mid}.json")
        )
        for mid in item["miner_ids"]
    ]
    pool = project_agent_artifact(bronze, origin="bronze")
    for m in miners:
        pool += project_agent_artifact(m.artifact, origin="miner", miner_id=m.miner_id)
    if any(c.paper_id not in (None, item["paper_id"]) for c in pool):
        raise ValueError("Candidate paper mismatch")
    merged = source_map(
        [source] + [read(folder / f"source_{m.miner_id}.json") for m in miners]
    )
    return bronze, source, miners, pool, merged, source_map([source])


def replay(item, args, config, verbose_paper):
    pid = item["paper_id"]
    out = args.output_dir / pid
    bronze, source, miners, pool, merged, eligibility = inputs(
        item, args.manifest.parent
    )
    collector = ModelUsageCollector(
        network="mainnet_replay",
        run_id=args.output_dir.name,
        batch_id=pid,
        checkpoint_sink=lambda events: write(out / "usage_events.json", events),
        checkpoint_every=1,
    )
    cfg = replace(
        config,
        root=out / "workspaces",
        eligibility_output_contract="verbose" if pid == verbose_paper else "compact",
        equivalence_group_bypass=pid != verbose_paper,
    )
    workflow = FileAgentSilverWorkflow(cfg, usage_sink=collector.record)
    report = {
        "paper_id": pid,
        "miner_count": len(miners),
        "candidate_count": len(pool),
        "output_contract": cfg.eligibility_output_contract,
    }
    try:
        result = run_paper_silver_pipeline(
            paper_id=pid,
            bronze_artifact=bronze,
            miner_artifacts=miners,
            silver_record_id="mainnet_replay_" + pid,
            adjudication_passes=[],
            paper_context=source.get("paper") or bronze.get("paper") or {},
            source_context_by_span_id=merged,
            eligibility_source_context_by_span_id=eligibility,
            file_agent_workflow=workflow,
            max_eligible_claims_per_miner=10,
            max_adjudication_cases=1000,
            filter_by_assessment=False,
        )
        record = result.silver_record.model_dump(mode="json")
        write(out / "silver_record.json", record)
        write(
            out / "decisions.json",
            [
                d.model_dump(mode="json")
                for d in result.eligibility_adjudication_decisions
            ],
        )
        write(out / "scores.json", [s.model_dump(mode="json") for s in result.scores])
        sessions = list((out / "workspaces").rglob("manifest.json"))
        session_root = sessions[0].parent if len(sessions) == 1 else None
        groups = []
        if (
            session_root
            and (session_root / "comparison/equivalence_groups.json").exists()
        ):
            from validator.agent_v1.comparison_models import EquivalenceGroup

            groups = [
                EquivalenceGroup.model_validate(g)
                for g in read(session_root / "comparison/equivalence_groups.json")[
                    "groups"
                ]
            ]
        before = read(
            args.manifest.parent / "historical/silver" / pid / "silver_record.json"
        )
        bset, aset = set(retained(before)), set(retained(record))
        report.update(
            status="complete",
            case_count=len(result.diff_cases),
            stage_timings=result.stage_timings,
            groups=[g.model_dump(mode="json") for g in groups],
            candidate_diffs=[
                {
                    "candidate_id": c.candidate_id,
                    "statement": c.statement,
                    "before_retained": c.candidate_id in bset,
                    "after_retained": c.candidate_id in aset,
                }
                for c in pool
                if (c.candidate_id in bset) != (c.candidate_id in aset)
            ],
            reliability=reliability(record, session_root, merged, groups)
            if session_root
            else {"error": "session root not unique"},
        )
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
    events = collector.snapshot()
    write(out / "usage_events.json", events)
    report["usage"] = usage_summary(events)
    write(out / "report.json", report)
    print(pid, report["status"], report["usage"].get("all", {}), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--env-file", type=Path)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    args = p.parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        p.error("Use a new output directory")
    items = read(args.manifest)
    if not items or len({i["paper_id"] for i in items}) != len(items):
        p.error("Distinct nonempty papers required")
    args.output_dir.mkdir(parents=True)
    checked = []
    for item in items:
        b, s, m, pool, merged, elig = inputs(item, args.manifest.parent)
        checked.append(
            {
                "paper_id": item["paper_id"],
                "miner_count": len(m),
                "candidate_count": len(pool),
                "source_spans": len(merged),
                "eligibility_source_spans": len(elig),
                "unresolved_candidate_spans": sorted(
                    {sid for c in pool for sid in c.source_span_ids} - merged.keys()
                ),
            }
        )
    write(args.output_dir / "input_validation.json", checked)
    if args.validate_only:
        return 0
    if not args.env_file:
        p.error("--env-file required for live replay")
    load_dotenv(args.env_file, override=True)
    # This runner never constructs a validator, wallet, backend client or subtensor.
    os.environ["CLAIMS_BACKEND_URL"] = ""
    os.environ["CLAIMS_AUDIT_ONLY"] = "true"
    disable_response_cache()
    cfg = replace(
        FileAgentWorkflowConfig.from_env(),
        equivalence_group_bypass=True,
        eligibility_sample_verbose_paper=False,
        resume_existing_stages=False,
        fallback_to_legacy=False,
        adjudication_max_workers=4,
    )
    verbose = min(
        items, key=lambda i: hashlib.sha256(i["paper_id"].encode()).hexdigest()
    )["paper_id"]
    status = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "papers": [],
        "verbose_paper": verbose,
        "workers": args.workers,
        "settings": {
            k: getattr(cfg, k)
            for k in (
                "harness",
                "canonicalization_harness",
                "comparison_model",
                "canonicalization_model",
                "canonical_audit_model",
                "adjudication_harness",
                "adjudication_max_tokens",
                "max_tokens",
                "adjudication_batch_size",
                "adjudication_max_workers",
            )
        },
        "scope": "comparison + eligibility + canonicalization; saved extraction/reference/diagnostic inputs; one verbose paper replaces compact; no chain/backend writes; response/stage caches disabled",
    }
    write(args.output_dir / "suite_status.json", status)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(replay, i, args, cfg, verbose): i["paper_id"] for i in items
        }
        for f in as_completed(futures):
            try:
                status["papers"].append(f.result())
            except Exception:
                status["papers"].append(
                    {
                        "paper_id": futures[f],
                        "status": "failed",
                        "traceback": traceback.format_exc(),
                    }
                )
            write(args.output_dir / "suite_status.json", status)
    status.update(
        status="complete"
        if all(i["status"] == "complete" for i in status["papers"])
        else "failed",
        finished_at=datetime.now(timezone.utc).isoformat(),
    )
    write(args.output_dir / "suite_status.json", status)
    return 0 if status["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
