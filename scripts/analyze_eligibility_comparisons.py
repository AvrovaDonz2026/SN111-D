"""Compare saved verbose/compact judge choices without making model requests."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def span_ids(value):
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"cited_span_ids", "reviewed_span_ids", "span_ids"}:
                found.update(child or [])
            else:
                found.update(span_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(span_ids(child))
    return found


def analyze(report, directory):
    modes = report["modes"]
    indexed = {m: {d["case_id"]: d for d in modes[m]["decisions"]} for m in ["verbose", "compact"]}
    fixture = json.loads((directory / "fixture.json").read_text())
    valid_spans = set(fixture["source_spans"])
    candidates = {c["candidate_id"]: c for b in fixture["contexts"] for c in b["candidates"]}
    rows = []
    counts = Counter()
    for cid in sorted(indexed["verbose"].keys() | indexed["compact"].keys()):
        left, right = (indexed[m].get(cid) for m in ["verbose", "compact"])
        counts["cases_in_union"] += 1
        if left is None or right is None:
            counts["missing_case_results"] += 1
            rows.append({"paper_id": report["paper_id"], "case_id": cid, "missing_case_result": True, "verbose": left, "compact": right})
            continue
        comparable = all(d.get("resolution_status") == "decided" for d in [left, right])
        changed = comparable and left["selected_candidate_id"] != right["selected_candidate_id"]
        counts["jointly_decided_cases"] += comparable
        counts["final_selection_changes"] += changed
        counts["route_changes"] += left["consensus_route"] != right["consensus_route"]
        item = {"paper_id": report["paper_id"], "case_id": cid, "jointly_decided": comparable,
                "final_selection_changed": changed, "verbose": left, "compact": right,
                "candidate_statements": {k: candidates[k]["statement"] for k in set(left.get("rejected_candidate_ids", [])) | set(right.get("rejected_candidate_ids", [])) | {x for x in [left["selected_candidate_id"], right["selected_candidate_id"]] if x} if k in candidates}, "judges": {}}
        for role in ["negative", "positive", "tiebreak"]:
            votes = []
            for d in [left, right]:
                votes.append(d.get("tiebreak_vote") if role == "tiebreak" else next((v for v in d.get("primary_votes", []) if v["judge_role"] == role), None))
            a, b = votes
            comparable_votes = a is not None and b is not None and not a.get("technical_unresolved") and not b.get("technical_unresolved")
            vote_changed = comparable_votes and a["selected_candidate_id"] != b["selected_candidate_id"]
            counts[role + "_comparable_votes"] += comparable_votes
            counts[role + "_selection_changes"] += vote_changed
            counts[role + "_execution_presence_changes"] += (a is None) != (b is None)
            item["judges"][role] = {"comparable": comparable_votes, "selection_changed": vote_changed,
                "execution_presence_changed": (a is None) != (b is None),
                "verbose": a, "compact": b}
        for mode, decision in [("verbose", left), ("compact", right)]:
            unknown = sorted(span_ids(decision) - valid_spans)
            item[mode + "_unknown_span_ids"] = unknown
            counts[mode + "_unknown_span_ids"] += len(unknown)
        rows.append(item)
    usage = {}
    for mode in ["verbose", "compact"]:
        events = json.loads((directory / mode / "usage_events.json").read_text())
        usage[mode] = {k: sum(e.get(k) or 0 for e in events) for k in ["prompt_tokens", "completion_tokens", "total_tokens"]}
        usage[mode].update(events=len(events), failed_events=sum(e["status"] == "failed" for e in events),
                           unknown_total_events=sum(e.get("total_tokens") is None for e in events))
    return {"paper_id": report["paper_id"], "fixture_sha256": report["fixture_sha256"],
            "mode_statuses": {m: modes[m]["status"] for m in modes}, "counts": dict(counts), "usage": usage, "cases": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    papers = [analyze(json.loads((p / "comparison.json").read_text()), p) for p in args.directories]
    counts = Counter()
    usage = {m: Counter() for m in ["verbose", "compact"]}
    for paper in papers:
        counts.update(paper["counts"])
        for m in usage:
            usage[m].update(paper["usage"][m])
    summary = {"paper_count": len(papers), "counts": dict(counts), "usage": {m: dict(v) for m, v in usage.items()}, "papers": papers}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "decision_comparison.json").write_text(json.dumps(summary, indent=2))
    columns = ["paper_id", "case_id", "judge", "comparable", "selection_changed", "verbose_selection", "compact_selection", "verbose_reason", "compact_reason", "verbose_span_ids", "compact_span_ids"]
    with (args.output_dir / "judge_decisions.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for paper in papers:
            for case in paper["cases"]:
                for judge, comparison in case.get("judges", {}).items():
                    a, b = comparison["verbose"], comparison["compact"]
                    writer.writerow({"paper_id": paper["paper_id"], "case_id": case["case_id"], "judge": judge,
                        "comparable": comparison["comparable"], "selection_changed": comparison["selection_changed"],
                        "verbose_selection": a.get("selected_candidate_id") if a else "NOT_EXECUTED",
                        "compact_selection": b.get("selected_candidate_id") if b else "NOT_EXECUTED",
                        "verbose_reason": a.get("rationale", "") if a else "", "compact_reason": b.get("rationale", "") if b else "",
                        "verbose_span_ids": json.dumps(sorted(span_ids(a))), "compact_span_ids": json.dumps(sorted(span_ids(b)))})
    print(json.dumps({"paper_count": len(papers), "counts": dict(counts), "usage": {m: dict(v) for m, v in usage.items()}}, indent=2))


if __name__ == "__main__":
    main()
