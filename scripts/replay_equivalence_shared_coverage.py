"""Offline experiment: fractional coverage for retained equivalence-group members only.

This does not change production scoring or execute models, wallets or backend clients.
Merged comparison groups share one final-unit pool. By default, group members
remain fractional even when they also have another contribution to the same unit.
Nongroup-only contributors retain full credit. An explicit max-credit variant
measures the sensitivity to this mixed-contributor choice.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scripts.benchmark_equivalence_routing import read, write
from validator.agent_v1.comparison_models import SilverRecord
from validator.agent_v1.record_projection import project_agent_artifact
from validator.agent_v1.silver_scoring import (
    IMPORTANCE_WEIGHTS,
    MINOR_TIER_CAP,
    _scored_units,
    _silver_score,
)


def unit_credit(unit, groups, owner, mixed_policy="group-priority"):
    """Return credit by miner and independently inspectable provenance."""
    retained = set(unit.equivalent_candidate_ids)
    if retained - owner.keys():
        raise ValueError("Final candidate owner is unknown")
    group_ids = unit.metadata.get("comparison_equivalence_group_ids", [])
    if set(group_ids) - groups.keys():
        raise ValueError("Unknown approved comparison group")
    expected = {g for g, members in groups.items() if retained.intersection(members)}
    if set(group_ids) != expected:
        raise ValueError("Retained group membership does not match unit provenance")
    members = set().union(*(set(groups[g]) for g in group_ids)) if group_ids else set()
    eligible = retained & members
    grouped_miners = {owner[c] for c in eligible if owner[c] is not None}
    independent = {owner[c] for c in retained - eligible if owner[c] is not None}
    n = len(grouped_miners)
    credits = {m: 1 / n for m in grouped_miners} if n else {}
    if mixed_policy not in {"max-credit", "group-priority"}:
        raise ValueError("Unknown mixed-contributor policy")
    full_credit = (
        independent if mixed_policy == "max-credit" else independent - grouped_miners
    )
    credits.update({m: 1.0 for m in full_credit})
    return credits, {
        "silver_unit_id": unit.silver_unit_id,
        "group_ids": group_ids,
        "retained_group_candidates": sorted(eligible),
        "group_miners": sorted(grouped_miners),
        "n": n,
        "independent_miners": sorted(independent),
        "full_credit_overrides": sorted(full_credit & grouped_miners),
        "mixed_policy": mixed_policy,
        "credit_by_miner": credits,
    }


def fractional_coverage(units, credits, miner_id):
    """Keep the existing importance denominator and minor-tier cap unchanged."""
    major_total = minor_total = major_credit = minor_credit = 0.0
    for u in units:
        weight = IMPORTANCE_WEIGHTS[u.importance]
        credit = credits[u.silver_unit_id].get(miner_id, 0.0)
        if not 0 <= credit <= 1:
            raise ValueError("Coverage credit outside [0, 1]")
        if u.importance == "minor":
            minor_total += weight
            minor_credit += weight * credit
        else:
            major_total += weight
            major_credit += weight * credit
    if major_total <= 0:
        return 1.0 if minor_total <= 0 else round(minor_credit / minor_total, 4)
    cap = MINOR_TIER_CAP / (1 - MINOR_TIER_CAP) * major_total
    return round(
        (major_credit + min(minor_credit, cap)) / (major_total + min(minor_total, cap)),
        4,
    )


def save_csv(path, rows):
    with path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def run(inputs, replay, original_analysis, output, mixed_policy="group-priority"):
    if output.exists():
        raise ValueError("Use a new experiment output directory")
    output.mkdir(parents=True)
    manifest = read(inputs / "manifest.json")
    base = {
        (s["paper_id"], s["miner_id"]): s
        for s in read(original_analysis / "corrected_replay_scores.json")
    }
    per_paper = {}
    audit = []
    ordinary_checks = []
    for item in manifest:
        pid = item["paper_id"]
        folder = replay / pid
        report = read(folder / "report.json")
        if report["status"] != "complete":
            continue
        record = SilverRecord.model_validate(read(folder / "silver_record.json"))
        owner = {
            c.candidate_id: None
            for c in project_agent_artifact(
                read(inputs / item["fixture_dir"] / "bronze.json"), origin="bronze"
            )
        }
        for mid in item["miner_ids"]:
            for c in project_agent_artifact(
                read(inputs / item["fixture_dir"] / f"miner_{mid}.json"),
                origin="miner",
                miner_id=mid,
            ):
                owner[c.candidate_id] = mid
        groups = {
            g["group_id"]: g["member_candidate_ids"] for g in report.get("groups", [])
        }
        if len(groups) != len(report.get("groups", [])):
            raise ValueError("Duplicate group IDs")
        credits = {}
        ordinary = {}
        for unit in _scored_units(record.silver_units):
            credits[unit.silver_unit_id], detail = unit_credit(
                unit, groups, owner, mixed_policy=mixed_policy
            )
            audit.append({"paper_id": pid, **detail})
            ordinary[unit.silver_unit_id] = {
                owner[c]: 1.0
                for c in unit.equivalent_candidate_ids
                if owner[c] is not None
            }
        per_paper[pid] = {}
        for mid in item["miner_ids"]:
            b = base[(pid, mid)]
            check = fractional_coverage(
                _scored_units(record.silver_units), ordinary, mid
            )
            if check != b["coverage"]:
                raise ValueError(f"Ordinary coverage did not reproduce {pid} {mid}")
            ordinary_checks.append((pid, mid))
            coverage = fractional_coverage(
                _scored_units(record.silver_units), credits, mid
            )
            empty = any(
                f.get("metadata", {}).get("code") == "empty_silver_record"
                for f in b["findings"]
            )
            score = _silver_score(
                coverage=coverage, quality=b["quality"], empty_silver_record=empty
            )
            if score > b["score"]:
                raise ValueError("Shared credit increased a fixed-Silver score")
            per_paper[pid][mid] = {
                "shared_coverage": coverage,
                "shared_score": score,
                "quality": b["quality"],
            }
    rows = []
    with (original_analysis / "miner_paper_score_changes.csv").open() as f:
        for old in csv.DictReader(f):
            pid, mid = old["paper_id"], old["miner_id"]
            new = per_paper[pid].get(mid)
            if new is None:
                if old["missing_submission"] != "True":
                    raise ValueError("Missing replay score")
                new = {"shared_coverage": 0.0, "shared_score": 0.0, "quality": 0.0}
            ordinary = float(old["new_score"])
            historical = float(old["old_score"])
            rows.append(
                {
                    "paper_id": pid,
                    "miner_id": mid,
                    "historical_score": historical,
                    "ordinary_replay_score": ordinary,
                    "ordinary_coverage": float(old["new_coverage"]),
                    **new,
                    "delta_vs_ordinary": round(new["shared_score"] - ordinary, 4),
                    "delta_vs_historical": round(new["shared_score"] - historical, 4),
                }
            )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["miner_id"]].append(row)
    miners = []
    for mid, rr in grouped.items():
        miners.append(
            {
                "miner_id": mid,
                "papers": len(rr),
                **{
                    f"{label}_mean": round(mean(r[field] for r in rr), 4)
                    for label, field in [
                        ("historical", "historical_score"),
                        ("ordinary", "ordinary_replay_score"),
                        ("shared", "shared_score"),
                    ]
                },
                "papers_reduced_vs_ordinary": sum(
                    r["shared_score"] < r["ordinary_replay_score"] for r in rr
                ),
            }
        )
    for label in ["historical", "ordinary", "shared"]:
        for n, m in enumerate(
            sorted(miners, key=lambda x: (-x[f"{label}_mean"], x["miner_id"])), 1
        ):
            m[f"{label}_rank"] = n
    miners.sort(key=lambda m: m["shared_rank"])
    summary = {
        "status": "complete",
        "policy": "equivalence members only; merged final-unit pool; nongroup-only miners full credit; mixed policy "
        + mixed_policy,
        "papers": len(per_paper),
        "miner_paper_slots": len(rows),
        "ordinary_scores_reproduced": len(ordinary_checks),
        "unit_count": len(audit),
        "group_units": sum(bool(u["group_ids"]) for u in audit),
        "groups_with_nonempty_miner_pool": sum(u["n"] > 0 for u in audit),
        "units_with_independent_contributions": sum(
            bool(u["group_ids"]) and bool(u["independent_miners"]) for u in audit
        ),
        "units_with_full_credit_override": sum(
            bool(u["full_credit_overrides"]) for u in audit
        ),
        "miner_paper_scores_reduced": sum(
            r["shared_score"] < r["ordinary_replay_score"] for r in rows
        ),
        "miner_summary": miners,
        "quality_and_silver_fixed": True,
        "token_usage_change": 0,
        "production_scoring_changed": False,
    }
    write(output / "summary.json", summary)
    write(output / "unit_credit_audit.json", audit)
    save_csv(output / "miner_summary.csv", miners)
    save_csv(output / "paper_scores.csv", rows)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["inputs", "replay", "original-analysis", "output-dir"]:
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument(
        "--mixed-contributor-policy",
        choices=["max-credit", "group-priority"],
        default="group-priority",
    )
    a = p.parse_args()
    print(
        run(
            a.inputs,
            a.replay,
            a.original_analysis,
            a.output_dir,
            mixed_policy=a.mixed_contributor_policy,
        )
    )


if __name__ == "__main__":
    main()
