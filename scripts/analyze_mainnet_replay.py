"""Compare frozen historical mainnet usage and decisions with a Silver replay."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

from scripts.benchmark_equivalence_routing import read, retained, usage_summary, write


def measured(events):
    return [
        e
        for e in events
        if e["stage_key"].startswith(
            ("silver_comparison", "silver_eligibility", "silver_canonical")
        )
    ]


def accounting(events):
    counts = usage_summary(events)
    issues = [
        {
            k: e.get(k)
            for k in (
                "paper_id",
                "stage_key",
                "status",
                "usage_event_id",
                "total_tokens",
                "completion_tokens",
            )
        }
        for e in events
        if e.get("total_tokens") is None
        or e.get("completion_tokens") is None
        or (e.get("status") == "success" and not e.get("total_tokens"))
    ]
    return {
        "counts": counts,
        "measurement_issues": issues,
        "known_usage_is_lower_bound": bool(issues),
    }


def verdicts(decisions):
    """Key by reviewed candidate set, independent of comparison case IDs."""
    result = defaultdict(list)
    for d in decisions:
        case = d.get("case") or {}
        ids = case.get("candidate_ids") or []
        if not ids:
            ids = [
                c.get("candidate_id")
                for c in case.get("candidates", [])
                if c.get("candidate_id")
            ]
        if not ids:
            ids = [*(d.get("rejected_candidate_ids") or [])]
            if d.get("selected_candidate_id"):
                ids.append(d["selected_candidate_id"])
        if ids:
            result[tuple(sorted(set(ids)))].append(d)
    return result


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(inputs, replay, out):
    out.mkdir(parents=True, exist_ok=True)
    status = read(replay / "suite_status.json")
    manifest = read(inputs / "manifest.json")
    historical = measured(read(inputs / "historical/usage_events.json"))
    ids = [x["usage_event_id"] for x in historical]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate historical usage event IDs")
    papers = []
    candidate_rows = []
    decision_rows = []
    group_rows = []
    after_events = []
    matched_before = []
    for item in manifest:
        pid = item["paper_id"]
        folder = replay / pid
        rp = folder / "report.json"
        if not rp.exists():
            continue
        report = read(rp)
        old_events = [e for e in historical if e.get("paper_id") == pid]
        new_events = (
            measured(read(folder / "usage_events.json"))
            if (folder / "usage_events.json").exists()
            else []
        )
        after_events += new_events
        row = {
            "paper_id": pid,
            "status": report["status"],
            "output_contract": report.get("output_contract"),
            "miner_count": len(item["miner_ids"]),
            "group_count": len(report.get("groups", [])),
            "before_cases": None,
            "after_cases": report.get("case_count"),
            "candidate_membership_changes": None,
            "before_output": sum(e.get("completion_tokens") or 0 for e in old_events),
            "after_output": sum(e.get("completion_tokens") or 0 for e in new_events),
            "before_total": sum(e.get("total_tokens") or 0 for e in old_events),
            "after_total": sum(e.get("total_tokens") or 0 for e in new_events),
            "before_unknown_events": len(accounting(old_events)["measurement_issues"]),
            "after_unknown_events": len(accounting(new_events)["measurement_issues"]),
        }
        if report["status"] == "complete":
            matched_before += old_events
            old = read(inputs / "historical/silver" / pid / "silver_record.json")
            new = read(folder / "silver_record.json")
            om = old["metadata"].get("eligibility_selection_adjudication", {})
            row["before_cases"] = om.get("case_count")
            ob = set(retained(old))
            nb = set(retained(new))
            row["candidate_membership_changes"] = len(ob ^ nb)
            for cid in sorted(ob ^ nb):
                candidate_rows.append(
                    {
                        "paper_id": pid,
                        "candidate_id": cid,
                        "before_retained": cid in ob,
                        "after_retained": cid in nb,
                    }
                )
            ov = verdicts(om.get("decisions", []))
            nv = verdicts(read(folder / "decisions.json"))
            for key in sorted(ov.keys() & nv.keys()):
                if len(ov[key]) != 1 or len(nv[key]) != 1:
                    continue
                a, b = ov[key][0], nv[key][0]
                decision_rows.append(
                    {
                        "paper_id": pid,
                        "candidate_ids": "|".join(key),
                        "before_selected": a.get("selected_candidate_id"),
                        "after_selected": b.get("selected_candidate_id"),
                        "changed": a.get("selected_candidate_id")
                        != b.get("selected_candidate_id"),
                        "before_route": a.get("consensus_route"),
                        "after_route": b.get("consensus_route"),
                        "before_reason": a.get("rationale") or a.get("reason"),
                        "after_reason": b.get("rationale") or b.get("reason"),
                    }
                )
            rel = report.get("reliability", {})
            row.update(
                canonical_audit_repaired=rel.get("audit_repaired"),
                canonical_draft_repaired=rel.get("draft_repaired"),
                unknown_final_span_ids=len(rel.get("unknown_final_span_ids", [])),
                unaccounted_group_members=sum(
                    len(g.get("unaccounted", [])) for g in rel.get("group_outcomes", [])
                ),
            )
            for group in rel.get("group_outcomes", []):
                group_rows.append({"paper_id": pid, **group})
        papers.append(row)
    complete = [p for p in papers if p["status"] == "complete"]
    summary = {
        "status": status["status"],
        "expected_papers": len(manifest),
        "reported_papers": len(papers),
        "completed_papers": len(complete),
        "historical_all_papers": accounting(historical),
        "replay_all_attempts": accounting(after_events),
        "matched_complete_historical": accounting(matched_before),
        "matched_complete_replay": accounting(
            [
                e
                for e in after_events
                if e.get("paper_id") in {p["paper_id"] for p in complete}
            ]
        ),
        "jointly_keyed_remaining_decisions": len(decision_rows),
        "jointly_keyed_decisions_changed": sum(d["changed"] for d in decision_rows),
        "candidate_membership_changes": len(candidate_rows),
        "papers": papers,
        "scope": "historical vs replay, not randomized causal estimate; provider-reported totals preserved; missing counters not imputed; embeddings/extraction/reference/diagnostics excluded",
    }
    write(out / "analysis.json", summary)
    write(out / "group_outcomes.json", group_rows)
    csv_write(out / "papers.csv", papers)
    csv_write(out / "candidate_membership_changes.csv", candidate_rows)
    csv_write(out / "remaining_decisions.csv", decision_rows)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--replay", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    summary = analyze(a.inputs, a.replay, a.output_dir)
    print(
        {
            k: v
            for k, v in summary.items()
            if k
            in (
                "status",
                "completed_papers",
                "reported_papers",
                "expected_papers",
                "jointly_keyed_remaining_decisions",
                "jointly_keyed_decisions_changed",
                "candidate_membership_changes",
            )
        }
    )


if __name__ == "__main__":
    main()
