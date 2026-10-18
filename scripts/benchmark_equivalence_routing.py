"""Paired live-model Silver replays of saved testnet artifacts; no backend/chain writes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import traceback
from types import SimpleNamespace

from dotenv import load_dotenv

from validator.agent_v1.comparison_dspy import DSPyComparisonRuntime
from validator.agent_v1.file_agent_workflow import (
    FileAgentWorkflowConfig,
    FileAgentWorkflowSession,
)
from validator.agent_v1.model_usage import ModelUsageCollector
from validator.agent_v1.orchestrator import (
    MinerArtifactSubmission,
    run_paper_silver_pipeline,
)
from validator.agent_v1.record_projection import project_agent_artifact


def disable_response_cache():
    # DSPy cache hits clear usage and would make a repeated arm appear free.
    # This affects only this benchmark process, never other VM services.
    import dspy

    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def usage_summary(events):
    groups = defaultdict(
        lambda: dict(
            events=0,
            failed_events=0,
            unknown_usage_events=0,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            reasoning_tokens=0,
        )
    )
    for event in events:
        key = event["stage_key"]
        component = (
            "comparison"
            if "comparison" in key
            else "canonicalization"
            if "canonical" in key
            else "eligibility"
        )
        for group in [groups[component], groups["all"]]:
            group["events"] += 1
            group["failed_events"] += event["status"] == "failed"
            group["unknown_usage_events"] += event.get("total_tokens") is None
            for field in [
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "reasoning_tokens",
            ]:
                group[field] += event.get(field) or 0
    return dict(groups)


class LiveSession(FileAgentWorkflowSession):
    """Use existing typed DSPy comparison runtime instead of a CLI agent.

    All routing, output validation, eligibility and canonical audit logic remains
    in the production workflow. Persist raw responses and failed usage too.
    """

    def _run_stage(self, **kwargs):
        if not kwargs["stage_key"].startswith("comparison"):
            return super()._run_stage(**kwargs)
        key = kwargs["stage_key"]
        folder = self.root / "executions" / key
        task = {
            **kwargs["task"],
            "skill_instructions": kwargs["skill_path"].read_text(),
        }
        self._atomic_json(folder / "task.json", task)
        self._atomic_json(
            folder / "output_schema.json", kwargs["output_model"].model_json_schema()
        )
        payload = DSPyComparisonRuntime(
            provider=self.config.adjudication_provider,
            api_base=self.config.adjudication_api_base,
            api_key_env=self.config.adjudication_api_key_env,
            max_tokens=self.config.adjudication_max_tokens,
            timeout_seconds=self.config.adjudication_timeout_seconds,
            usage_sink=self.usage_sink,
            raw_output_sink=lambda raw: self._atomic_json(
                folder / "raw_response.json", {"raw_response": raw}
            ),
        ).run(
            task=task,
            output_model=kwargs["output_model"],
            model=kwargs["model"],
            stage_key=key,
            paper_id=self.paper_id,
            workspace_id=self.workspace_id,
        )
        self._atomic_json(folder / "output.json", payload.model_dump(mode="json"))
        self._record_manifest_stage(
            {"stage_key": key, "status": "complete", "harness": "dspy"}
        )
        return SimpleNamespace(payload=payload)


class FrozenComparisonSession(LiveSession):
    def run_comparison(self):
        self.confirmed_equivalence_groups = (
            self.frozen_groups if self.config.equivalence_group_bypass else []
        )
        self._write_json(
            "comparison/equivalence_groups.json",
            {
                "schema": "claims_comparison_equivalence_groups_v1",
                "paper_id": self.paper_id,
                "workspace_id": self.workspace_id,
                "groups": [
                    g.model_dump(mode="json") for g in self.confirmed_equivalence_groups
                ],
            },
        )
        self._write_json(
            "comparison/comparison_pairs.json",
            {
                "edges": [e.model_dump(mode="json") for e in self.frozen_edges],
                "comparison_source": "shared_live_comparison",
                "model_calls_reused_between_arms": True,
            },
        )
        return self.frozen_edges


class Workflow:
    def __init__(self, config, collector, edges, groups):
        self.config, self.collector, self.edges, self.groups = (
            config,
            collector,
            edges,
            groups,
        )

    def start_session(self, **kwargs):
        self.session = FrozenComparisonSession(
            config=self.config, usage_sink=self.collector.record, **kwargs
        )
        self.session.frozen_edges, self.session.frozen_groups = self.edges, self.groups
        return self.session


def retained(record):
    return sorted(
        {
            cid
            for unit in record["silver_units"]
            for cid in unit["equivalent_candidate_ids"]
        }
    )


def reliability(record, session_root, source_map, groups):
    manifest = read(session_root / "manifest.json")
    metadata = record.get("metadata", {}).get("file_agent_workflow", {})
    units = record["silver_units"]
    kept = set(retained(record))
    exclusions = metadata.get("canonical_exclusions", [])
    excluded = {cid for row in exclusions for cid in row["candidate_ids"]}
    group_rows = []
    for group in groups:
        members = set(group.member_candidate_ids)
        containing = [
            u["silver_unit_id"]
            for u in units
            if members.intersection(u["equivalent_candidate_ids"])
        ]
        group_rows.append(
            {
                "group_id": group.group_id,
                "members": sorted(members),
                "retained": sorted(members & kept),
                "excluded": sorted(members & excluded),
                "unaccounted": sorted(members - kept - excluded),
                "unit_ids": containing,
                "split": len(containing) > 1,
                "fully_excluded": members.issubset(excluded),
            }
        )
    draft_path = session_root / "silver/canonical_draft.json"
    draft = read(draft_path) if draft_path.exists() else {}
    return {
        "manifest_stages": manifest["stages"],
        "quality_checks": metadata.get("canonical_quality_checks"),
        "audit_repaired": metadata.get("canonical_audit_repaired"),
        "audit_initial_rejection": metadata.get("canonical_audit_initial_rejection"),
        "draft_repaired": metadata.get("canonical_draft_repaired"),
        "draft_initial_failure": metadata.get("canonical_draft_initial_failure"),
        "audit_findings": metadata.get("canonical_audit_findings", []),
        "draft_unit_count": len(draft.get("units", [])),
        "final_unit_count": len(units),
        "draft_exclusion_count": len(draft.get("exclusions", [])),
        "final_exclusions": exclusions,
        "unknown_final_span_ids": sorted(
            {sid for unit in units for sid in unit["source_span_ids"]} - set(source_map)
        ),
        "group_outcomes": group_rows,
    }


def fixture_inputs(item, manifest_dir):
    paper = item["paper_id"]
    folder = manifest_dir / item["fixture_dir"]
    bronze, source = read(folder / "bronze.json"), read(folder / "source.json")
    miners = [
        MinerArtifactSubmission(
            miner_id=mid, artifact=read(folder / f"miner_{mid}.json")
        )
        for mid in item["miner_ids"]
    ]
    candidates = project_agent_artifact(bronze, origin="bronze")
    for submission in miners:
        candidates += project_agent_artifact(
            submission.artifact, origin="miner", miner_id=submission.miner_id
        )
    source_map = {
        row["span_id"]: row["text"]
        for row in source["spans"]
        if row.get("span_id") and row.get("text")
    }
    evidence_spans = {sid for c in candidates for sid in c.source_span_ids}
    if evidence_spans - source_map.keys():
        raise ValueError(f"{paper}: unknown input source spans")
    if any(c.paper_id not in {None, paper} for c in candidates):
        raise ValueError(f"{paper}: candidate belongs to a different paper")
    return bronze, source, miners, candidates, source_map


def run_paper(item, index, args, config):
    paper = item["paper_id"]
    bronze, source, miners, candidates, source_map = fixture_inputs(
        item, args.manifest.parent
    )
    out = args.output_dir / paper
    out.mkdir(parents=True)
    fixture = {
        "paper_id": paper,
        "bronze": bronze,
        "miners": [asdict(m) for m in miners],
        "source": source,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fixture, sort_keys=True).encode()
    ).hexdigest()
    write(out / "fixture.json", fixture)
    report = {
        "paper_id": paper,
        "fixture_sha256": fingerprint,
        "candidate_count": len(candidates),
        "miner_count": len(miners),
        "arms": {},
        "comparison": {},
    }
    comparison_usage = ModelUsageCollector(
        network="testnet", run_id="equivalence_benchmark", batch_id=paper
    )
    common = LiveSession(
        config=replace(config, root=out / "workspaces", equivalence_group_bypass=True),
        paper_id=paper,
        workspace_id="shared_comparison",
        candidates=candidates,
        paper_context=source.get("paper") or bronze.get("paper") or {},
        source_context_by_span_id=source_map,
        usage_sink=comparison_usage.record,
    )
    try:
        edges = common.run_comparison()
        groups = common.confirmed_equivalence_groups
    except Exception:
        write(out / "comparison_usage_events.json", comparison_usage.snapshot())
        report["comparison"] = {"status": "failed", "traceback": traceback.format_exc()}
        write(out / "report.json", report)
        return report
    report["comparison"] = {
        "status": "complete",
        "edge_count": len(edges),
        "groups": [g.model_dump(mode="json") for g in groups],
        "usage": usage_summary(comparison_usage.snapshot()),
    }
    write(out / "comparison_usage_events.json", comparison_usage.snapshot())
    order = ["before", "after"] if index % 2 == 0 else ["after", "before"]
    report["arm_order"] = order
    for arm in order:
        collector = ModelUsageCollector(
            network="testnet",
            run_id=f"equivalence_{arm}",
            batch_id=paper,
            checkpoint_sink=lambda events, arm=arm: write(
                out / arm / "usage_events.json", events
            ),
            checkpoint_every=1,
        )
        arm_config = replace(
            config, root=out / "workspaces", equivalence_group_bypass=arm == "after"
        )
        workflow = Workflow(arm_config, collector, edges, groups)
        try:
            result = run_paper_silver_pipeline(
                paper_id=paper,
                bronze_artifact=bronze,
                miner_artifacts=miners,
                silver_record_id=arm,
                adjudication_passes=[],
                paper_context=source.get("paper") or bronze.get("paper") or {},
                source_context_by_span_id=source_map,
                eligibility_source_context_by_span_id=source_map,
                file_agent_workflow=workflow,
                max_eligible_claims_per_miner=100,
                max_adjudication_cases=200,
            )
            record = result.silver_record.model_dump(mode="json")
            write(out / arm / "silver_record.json", record)
            write(
                out / arm / "decisions.json",
                [
                    d.model_dump(mode="json")
                    for d in result.eligibility_adjudication_decisions
                ],
            )
            report["arms"][arm] = {
                "status": "complete",
                "case_count": len(result.diff_cases),
                "retained_candidate_ids": retained(record),
                "scores": [s.model_dump(mode="json") for s in result.scores],
                "stage_timings": result.stage_timings,
                "reliability": reliability(
                    record,
                    workflow.session.root,
                    source_map,
                    groups if arm == "after" else [],
                ),
            }
        except Exception:
            report["arms"][arm] = {
                "status": "failed",
                "traceback": traceback.format_exc(),
            }
        events = collector.snapshot()
        write(out / arm / "usage_events.json", events)
        report["arms"][arm]["usage_excluding_shared_comparison"] = usage_summary(events)
        report["arms"][arm]["usage_including_shared_comparison"] = usage_summary(
            events + comparison_usage.snapshot()
        )
        write(out / "report.json", report)
        print(paper, arm, report["arms"][arm]["status"], flush=True)
    if all(report["arms"][a]["status"] == "complete" for a in ["before", "after"]):
        before, after = (
            set(report["arms"][a]["retained_candidate_ids"])
            for a in ["before", "after"]
        )
        report["candidate_diffs"] = [
            {
                "candidate_id": c.candidate_id,
                "statement": c.statement,
                "miner_id": c.miner_id,
                "before_retained": c.candidate_id in before,
                "after_retained": c.candidate_id in after,
                "span_ids": c.source_span_ids,
            }
            for c in candidates
            if (c.candidate_id in before) != (c.candidate_id in after)
        ]
    write(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="deepseek/deepseek-v4-flash")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check all fixture identities and span references without model calls.",
    )
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--workers", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        parser.error("Use a new output directory.")
    load_dotenv(args.env_file, override=True)
    args.output_dir.mkdir(parents=True)
    items = read(args.manifest)[: args.limit]
    if not items or len({item["paper_id"] for item in items}) != len(items):
        parser.error("Use a nonempty manifest of distinct papers.")
    if args.validate_only:
        papers = []
        for item in items:
            bronze, source, miners, candidates, source_map = fixture_inputs(
                item, args.manifest.parent
            )
            papers.append(
                {
                    "paper_id": item["paper_id"],
                    "candidate_count": len(candidates),
                    "miner_count": len(miners),
                    "source_span_count": len(source_map),
                    "input_sha256": hashlib.sha256(
                        json.dumps(
                            {
                                "bronze": bronze,
                                "source": source,
                                "miners": [asdict(m) for m in miners],
                            },
                            sort_keys=True,
                        ).encode()
                    ).hexdigest(),
                }
            )
        write(
            args.output_dir / "input_validation.json",
            {"status": "validated", "model_calls": 0, "papers": papers},
        )
        print(
            json.dumps(
                {"status": "validated", "model_calls": 0, "papers": papers}, indent=2
            )
        )
        return 0
    disable_response_cache()
    base = FileAgentWorkflowConfig.from_env()
    config = replace(
        base,
        adjudication_harness="dspy",
        canonicalization_harness="dspy",
        comparison_model=args.model,
        canonicalization_model=args.model,
        canonical_audit_model=args.model,
        adjudication_negative_model=args.model,
        adjudication_positive_model=args.model,
        adjudication_tiebreak_model=args.model,
        eligibility_output_contract="compact",
        eligibility_sample_verbose_paper=False,
        adjudication_max_workers=2,
        adjudication_batch_size=4,
        adjudication_max_tokens=32768,
        adjudication_appellate_max_tokens=32768,
        adjudication_timeout_seconds=600,
        max_tokens=32768,
        timeout_seconds=600,
        fallback_to_legacy=False,
        resume_existing_stages=False,
    )
    status = {
        "schema": "claims_equivalence_routing_benchmark_v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "model": args.model,
        "response_cache": {"disk": False, "memory": False, "stage_resume": False},
        "settings": {
            key: getattr(config, key)
            for key in [
                "adjudication_provider",
                "adjudication_api_base",
                "adjudication_api_key_env",
                "canonicalization_provider",
                "canonicalization_api_base",
                "canonicalization_api_key_env",
                "eligibility_output_contract",
                "eligibility_sample_verbose_paper",
                "adjudication_batch_size",
                "adjudication_max_workers",
                "adjudication_max_tokens",
                "adjudication_appellate_max_tokens",
                "adjudication_timeout_seconds",
                "max_tokens",
                "timeout_seconds",
                "fallback_to_legacy",
            ]
        },
        "papers": [],
        "scope": "comparison + eligibility + canonicalization, saved testnet artifacts, no backend/chain writes",
        "shared_comparison": "one live comparison per paper charged logically to each arm; physical calls saved once",
        "embedding_usage": "Embedding retrieval usage is not exposed by the existing pairing provider; excluded from token totals.",
    }
    write(args.output_dir / "suite_status.json", status)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(run_paper, item, i, args, config): item["paper_id"]
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            try:
                status["papers"].append(future.result())
            except Exception:
                status["papers"].append(
                    {"paper_id": futures[future], "error": traceback.format_exc()}
                )
            write(args.output_dir / "suite_status.json", status)
    status["finished_at"] = datetime.now(timezone.utc).isoformat()
    status["status"] = (
        "complete"
        if all(
            p.get("arms") and all(a["status"] == "complete" for a in p["arms"].values())
            for p in status["papers"]
        )
        else "failed"
    )
    write(args.output_dir / "suite_status.json", status)
    return 0 if status["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
