from scripts.analyze_mainnet_replay import accounting, measured, verdicts


def test_historical_missing_counters_are_lower_bounds_not_imputed():
    events = [
        {
            "stage_key": "silver_eligibility_adjudication_positive",
            "status": "success",
            "prompt_tokens": 3,
            "completion_tokens": 2,
            "total_tokens": 5,
        },
        {
            "stage_key": "silver_eligibility_adjudication_negative",
            "status": "success",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
    ]
    result = accounting(events)
    assert result["known_usage_is_lower_bound"]
    assert result["counts"]["all"]["total_tokens"] == 5
    assert len(result["measurement_issues"]) == 1


def test_decisions_match_by_candidates_without_old_case_ids():
    a = {
        "case_id": "old",
        "selected_candidate_id": "miner:c1",
        "rejected_candidate_ids": ["bronze:c1"],
    }
    b = {
        "case_id": "new",
        "selected_candidate_id": None,
        "rejected_candidate_ids": ["miner:c1", "bronze:c1"],
    }
    assert set(verdicts([a])) == set(verdicts([b]))


def test_scope_excludes_reference_diagnostics_and_embeddings():
    keys = [
        "reference_miner",
        "diagnostic_validation",
        "embedding",
        "silver_comparison",
        "silver_eligibility_adjudication_negative",
        "silver_canonicalization_audit",
    ]
    assert [x["stage_key"] for x in measured([{"stage_key": k} for k in keys])] == keys[
        3:
    ]
