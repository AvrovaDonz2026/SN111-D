"""Summarize paired equivalence-routing replays, separating provenance gains from other changes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from itertools import combinations
import json
from pathlib import Path

from scripts.benchmark_equivalence_routing import usage_summary


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def analyze(root):
    root = Path(root)
    reports = [read(p) for p in sorted(root.glob("openalex*/report.json"))]
    measurement_issues = {}
    for report in reports:
        folder = root / report["paper_id"]
        issues = []
        for path in [
            folder / "comparison_usage_events.json",
            *[folder / a / "usage_events.json" for a in ["before", "after"]],
        ]:
            for event in read(path, []):
                if (
                    event.get("status") == "success"
                    and (event.get("total_tokens") or 0) <= 0
                ):
                    issues.append(
                        {
                            "file": str(path.relative_to(root)),
                            "stage": event["stage_key"],
                            "issue": "successful_model_event_has_no_positive_token_usage",
                        }
                    )
        if issues:
            measurement_issues[report["paper_id"]] = issues
    paired = [
        p
        for p in reports
        if p["paper_id"] not in measurement_issues
        and all(
            p.get("arms", {}).get(a, {}).get("status") == "complete"
            for a in ["before", "after"]
        )
    ]
    totals = {a: defaultdict(Counter) for a in ["before", "after"]}
    all_attempts = {a: [] for a in ["before", "after"]}
    physical = []
    rows, paper_rows, common_decisions, partitions, group_decisions = [], [], [], [], []
    reliability_counts = {a: Counter() for a in ["before", "after"]}
    for report in reports:
        folder = root / report["paper_id"]
        comparison = read(folder / "comparison_usage_events.json", [])
        physical += comparison
        for arm in ["before", "after"]:
            events = read(folder / arm / "usage_events.json", [])
            physical += events
            all_attempts[arm] += comparison + events
    for report in paired:
        paper = report["paper_id"]
        folder = root / paper
        group_by_member = {
            cid: g["group_id"]
            for g in report["comparison"].get("groups", [])
            for cid in g["member_candidate_ids"]
        }
        row = {
            "paper_id": paper,
            "group_count": len(report["comparison"].get("groups", [])),
            "grouped_candidates": len(group_by_member),
            "changed_group_members": 0,
            "changed_other_candidates": 0,
        }
        records, decisions = {}, {}
        for arm in ["before", "after"]:
            arm_report = report["arms"][arm]
            row[arm + "_cases"] = arm_report["case_count"]
            for component, values in arm_report[
                "usage_including_shared_comparison"
            ].items():
                totals[arm][component].update(values)
            r = arm_report["reliability"]
            counts = reliability_counts[arm]
            counts["completed_papers"] += 1
            counts["audit_repaired_papers"] += bool(r["audit_repaired"])
            counts["draft_repaired_papers"] += bool(r["draft_repaired"])
            counts["audit_findings"] += len(r["audit_findings"])
            counts["unknown_final_span_ids"] += len(r["unknown_final_span_ids"])
            counts["split_groups"] += sum(g["split"] for g in r["group_outcomes"])
            counts["fully_excluded_groups"] += sum(
                g["fully_excluded"] for g in r["group_outcomes"]
            )
            counts["unaccounted_group_members"] += sum(
                len(g["unaccounted"]) for g in r["group_outcomes"]
            )
            counts["canonical_candidates"] += sum(
                len(x["equivalent_candidate_ids"])
                for x in read(folder / arm / "silver_record.json")["silver_units"]
            ) + sum(len(e["candidate_ids"]) for e in r["final_exclusions"])
            row[arm + "_output_tokens"] = (
                arm_report["usage_including_shared_comparison"]
                .get("all", {})
                .get("completion_tokens", 0)
            )
            row[arm + "_total_tokens"] = (
                arm_report["usage_including_shared_comparison"]
                .get("all", {})
                .get("total_tokens", 0)
            )
            row[arm + "_canonical_output_tokens"] = (
                arm_report["usage_including_shared_comparison"]
                .get("canonicalization", {})
                .get("completion_tokens", 0)
            )
            row[arm + "_canonical_candidates"] = len(
                read(
                    folder
                    / "workspaces"
                    / arm
                    / paper
                    / "executions/canonicalization_draft/task.json",
                    {},
                ).get("accepted_candidates", [])
            )
            records[arm] = read(folder / arm / "silver_record.json")
            decisions[arm] = {
                d["case_id"]: d for d in read(folder / arm / "decisions.json", [])
            }
        for changed in report.get("candidate_diffs", []):
            grouped = changed["candidate_id"] in group_by_member
            row["changed_group_members" if grouped else "changed_other_candidates"] += 1
            rows.append(
                {
                    "paper_id": paper,
                    **changed,
                    "bypassed_group_member": grouped,
                    "group_id": group_by_member.get(changed["candidate_id"]),
                }
            )
        for cid in sorted(decisions["before"].keys() & decisions["after"].keys()):
            a, b = decisions["before"][cid], decisions["after"][cid]
            jointly_decided = (
                a.get("resolution_status", "decided")
                == b.get("resolution_status", "decided")
                == "decided"
            )
            common_decisions.append(
                {
                    "paper_id": paper,
                    "case_id": cid,
                    "jointly_decided": jointly_decided,
                    "before_selection": a["selected_candidate_id"],
                    "after_selection": b["selected_candidate_id"],
                    "selection_changed": jointly_decided
                    and a["selected_candidate_id"] != b["selected_candidate_id"],
                    "before_reason": a["rationale"],
                    "after_reason": b["rationale"],
                }
            )
        for group in report["comparison"].get("groups", []):
            members = set(group["member_candidate_ids"])
            kept = {
                arm: sorted(
                    members.intersection(report["arms"][arm]["retained_candidate_ids"])
                )
                for arm in ["before", "after"]
            }
            group_decisions.append(
                {
                    "paper_id": paper,
                    "group_id": group["group_id"],
                    "before_retained_members": kept["before"],
                    "after_retained_members": kept["after"],
                    "finding_present_before": bool(kept["before"]),
                    "finding_present_after": bool(kept["after"]),
                    "finding_acceptance_changed": bool(kept["before"])
                    != bool(kept["after"]),
                }
            )
        membership = {
            arm: {
                cid: i
                for i, u in enumerate(record["silver_units"])
                for cid in u["equivalent_candidate_ids"]
            }
            for arm, record in records.items()
        }
        shared = membership["before"].keys() & membership["after"].keys()
        for a, b in combinations(sorted(shared), 2):
            together = {
                arm: membership[arm][a] == membership[arm][b]
                for arm in ["before", "after"]
            }
            if together["before"] != together["after"]:
                partitions.append(
                    {
                        "paper_id": paper,
                        "candidate_a": a,
                        "candidate_b": b,
                        "before_together": together["before"],
                        "after_together": together["after"],
                    }
                )
        row["output_reduction_pct"] = percent_reduction(
            row["before_output_tokens"], row["after_output_tokens"]
        )
        row["total_reduction_pct"] = percent_reduction(
            row["before_total_tokens"], row["after_total_tokens"]
        )
        paper_rows.append(row)
    tokens = {
        arm: {component: dict(v) for component, v in components.items()}
        for arm, components in totals.items()
    }
    savings = {
        field: percent_reduction(
            tokens["before"].get("all", {}).get(field, 0),
            tokens["after"].get("all", {}).get(field, 0),
        )
        for field in ["completion_tokens", "total_tokens"]
    }
    return {
        "completed_paired_papers": len(paired),
        "reports_found": len(reports),
        "measurement_issues": measurement_issues,
        "paired_usage": tokens,
        "paired_reduction_pct": savings,
        "physical_model_usage": usage_summary(physical),
        "all_attempt_usage_logical": {
            arm: usage_summary(e) for arm, e in all_attempts.items()
        },
        "failures": [
            {
                "paper_id": p["paper_id"],
                "comparison_status": p.get("comparison", {}).get("status"),
                "arms": {a: v["status"] for a, v in p.get("arms", {}).items()},
            }
            for p in reports
            if p not in paired
        ],
        "canonical_reliability": {a: dict(v) for a, v in reliability_counts.items()},
        "papers": paper_rows,
        "candidate_diffs": rows,
        "unchanged_route_decisions": common_decisions,
        "shared_candidate_partition_changes": partitions,
        "group_decision_changes": group_decisions,
        "interpretation": "Membership changes inside bypassed groups can restore supported contributor provenance; neither agreement nor a completed audit establishes scientific correctness.",
    }


def percent_reduction(before, after):
    return round(100 * (before - after) / before, 3) if before else None


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, indent=2))
    for key in [
        "papers",
        "candidate_diffs",
        "unchanged_route_decisions",
        "shared_candidate_partition_changes",
        "group_decision_changes",
    ]:
        write_csv(args.output_dir / (key + ".csv"), result[key])
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "completed_paired_papers",
                    "paired_usage",
                    "paired_reduction_pct",
                    "canonical_reliability",
                    "failures",
                ]
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
