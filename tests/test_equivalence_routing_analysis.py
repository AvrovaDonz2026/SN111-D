import json

from scripts.analyze_equivalence_routing import analyze


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def event(stage, output, failed=False):
    return {
        "stage_key": stage,
        "status": "failed" if failed else "success",
        "prompt_tokens": 100,
        "completion_tokens": output,
        "total_tokens": 100 + output,
    }


def test_shared_comparison_counted_once_physically_and_equally_in_both_arms(tmp_path):
    folder = tmp_path / "openalex_paper"
    shared = [event("silver_comparison", 5)]
    save(folder / "comparison_usage_events.json", shared)
    arms = {}
    for arm, output, ids in [
        ("before", 20, ["bronze:b"]),
        ("after", 10, ["bronze:b", "miner:m"]),
    ]:
        save(
            folder / arm / "usage_events.json",
            [event("silver_canonicalization_audit", output)],
        )
        save(
            folder / arm / "silver_record.json",
            {"silver_units": [{"equivalent_candidate_ids": ids}]},
        )
        save(folder / arm / "decisions.json", [])
        arms[arm] = {
            "status": "complete",
            "case_count": int(arm == "before"),
            "retained_candidate_ids": ids,
            "usage_including_shared_comparison": {
                "all": {"completion_tokens": output + 5, "total_tokens": output + 205}
            },
            "reliability": {
                "audit_repaired": False,
                "draft_repaired": False,
                "audit_findings": [],
                "unknown_final_span_ids": [],
                "group_outcomes": [],
                "final_exclusions": [],
            },
        }
    save(
        folder / "report.json",
        {
            "paper_id": "openalex_paper",
            "arms": arms,
            "comparison": {
                "status": "complete",
                "groups": [
                    {"group_id": "g", "member_candidate_ids": ["bronze:b", "miner:m"]}
                ],
            },
            "candidate_diffs": [
                {
                    "candidate_id": "miner:m",
                    "before_retained": False,
                    "after_retained": True,
                }
            ],
        },
    )
    result = analyze(tmp_path)
    assert result["physical_model_usage"]["all"]["completion_tokens"] == 35
    assert result["paired_usage"]["before"]["all"]["completion_tokens"] == 25
    assert result["paired_usage"]["after"]["all"]["completion_tokens"] == 15
    assert result["cohorts"]["with_bypass"]["paper_count"] == 1
    assert result["cohorts"]["without_bypass"]["paper_count"] == 0
    assert result["papers"][0]["changed_group_members"] == 1
    assert result["papers"][0]["changed_other_candidates"] == 0
    assert result["group_decision_changes"][0]["finding_acceptance_changed"] is False


def test_failed_paper_costs_remain_visible_and_do_not_enter_paired_savings(tmp_path):
    folder = tmp_path / "openalex_failed"
    save(folder / "comparison_usage_events.json", [event("silver_comparison", 5)])
    save(
        folder / "after/usage_events.json",
        [event("silver_canonicalization_audit", 20, failed=True)],
    )
    save(
        folder / "report.json",
        {
            "paper_id": "openalex_failed",
            "comparison": {"status": "complete"},
            "arms": {"after": {"status": "failed"}},
        },
    )
    result = analyze(tmp_path)
    assert result["completed_paired_papers"] == 0
    assert result["paired_reduction_pct"]["completion_tokens"] is None
    assert result["physical_model_usage"]["all"]["completion_tokens"] == 25
    assert (
        result["all_attempt_usage_logical"]["after"]["canonicalization"][
            "failed_events"
        ]
        == 1
    )
    assert len(result["failures"]) == 1


def test_successful_cached_zero_usage_is_flagged_and_excluded(tmp_path):
    folder = tmp_path / "openalex_cached"
    save(folder / "comparison_usage_events.json", [event("silver_comparison", 5)])
    save(
        folder / "before/usage_events.json",
        [
            {
                "stage_key": "silver_canonicalization_audit",
                "status": "success",
                "total_tokens": 0,
                "completion_tokens": 0,
            }
        ],
    )
    save(
        folder / "report.json",
        {
            "paper_id": "openalex_cached",
            "comparison": {"status": "complete"},
            "arms": {arm: {"status": "complete"} for arm in ["before", "after"]},
        },
    )
    result = analyze(tmp_path)
    assert result["completed_paired_papers"] == 0
    assert (
        result["measurement_issues"]["openalex_cached"][0]["stage"]
        == "silver_canonicalization_audit"
    )
