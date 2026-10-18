import json
from pathlib import Path

import pytest

from scripts.benchmark_equivalence_routing import fixture_inputs, main, usage_summary


def fixture(tmp_path):
    paper = tmp_path / "paper"
    paper.mkdir()
    artifact = {
        "paper": {"paper_id": "paper"},
        "logic": {
            "claims": [
                {
                    "claim_id": "c1",
                    "statement": "A supported finding.",
                    "sources": [{"span_ids": ["s1"]}],
                }
            ]
        },
    }
    (paper / "bronze.json").write_text(json.dumps(artifact))
    (paper / "miner_uid_1.json").write_text(json.dumps(artifact))
    (paper / "source.json").write_text(
        json.dumps({"spans": [{"span_id": "s1", "text": "Source text."}]})
    )
    return {"paper_id": "paper", "fixture_dir": "paper", "miner_ids": ["uid_1"]}


def test_validate_only_checks_artifacts_without_loading_model_config(
    tmp_path, monkeypatch
):
    item = fixture(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([item]))
    env = tmp_path / "empty.env"
    env.touch()
    monkeypatch.setattr(
        "scripts.benchmark_equivalence_routing.FileAgentWorkflowConfig.from_env",
        lambda: pytest.fail("validate-only must not configure or execute live models"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "benchmark",
            "--manifest",
            str(manifest),
            "--env-file",
            str(env),
            "--output-dir",
            str(tmp_path / "validated"),
            "--validate-only",
        ],
    )
    assert main() == 0
    result = json.loads((tmp_path / "validated/input_validation.json").read_text())
    assert result["model_calls"] == 0
    assert result["papers"][0]["candidate_count"] == 2


def test_input_validation_rejects_source_ids_outside_paper(tmp_path):
    item = fixture(tmp_path)
    (tmp_path / "paper/source.json").write_text(json.dumps({"spans": []}))
    with pytest.raises(ValueError, match="unknown input source spans"):
        fixture_inputs(item, tmp_path)


def test_usage_keeps_failed_calls_and_reports_missing_usage():
    totals = usage_summary(
        [
            {
                "stage_key": "silver_canonicalization_audit",
                "status": "failed",
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            {
                "stage_key": "silver_eligibility_negative",
                "status": "success",
                "prompt_tokens": 50,
                "completion_tokens": 10,
                "total_tokens": 60,
            },
            {"stage_key": "silver_canonicalization_audit_repair", "status": "failed"},
        ]
    )
    assert totals["all"]["total_tokens"] == 180
    assert totals["all"]["completion_tokens"] == 30
    assert totals["canonicalization"]["failed_events"] == 2
    assert totals["canonicalization"]["unknown_usage_events"] == 1


def test_live_benchmark_disables_both_dspy_caches(monkeypatch):
    import sys
    from types import SimpleNamespace
    from scripts.benchmark_equivalence_routing import disable_response_cache

    observed = []
    monkeypatch.setitem(
        sys.modules,
        "dspy",
        SimpleNamespace(configure_cache=lambda **kwargs: observed.append(kwargs)),
    )
    disable_response_cache()
    assert observed == [{"enable_disk_cache": False, "enable_memory_cache": False}]
