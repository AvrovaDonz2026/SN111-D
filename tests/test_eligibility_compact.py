from dataclasses import replace
from types import MethodType
import json

import pytest

from validator.agent_v1.eligibility_compact import (
    CompactEligibilityOutput,
    CompactBlindOutput,
    candidate_findings,
    sampled_verbose_paper,
    validate_compact,
)
from validator.agent_v1.eligibility_dspy import _constrained_output_model
from test_validator_agent_v1_eligibility import _session, _candidate, _context


def _task(pair=True):
    return {
        "source_spans": {"S1": "Trial evidence."},
        "cases": [
            {
                "case_ref": "k0",
                "candidates": [{"candidate_ref": "k0_a", "source_span_ids": ["S1"]}]
                + (
                    [{"candidate_ref": "k0_b", "source_span_ids": ["S1"]}]
                    if pair
                    else []
                ),
            }
        ],
    }


@pytest.mark.parametrize(
    "decision,spans,reason,pair",
    [
        ("a", ["invented"], "Supported", True),
        ("a", [], "Supported", True),
        ("b", ["S1"], "Supported", False),
        ("a", ["S1"], "  ", True),
    ],
)
def test_invalid_compact_verdicts_are_rejected(decision, spans, reason, pair):
    payload = CompactEligibilityOutput(
        decisions={"k0": {"decision": decision, "reason": reason, "span_ids": spans}}
    )
    with pytest.raises(ValueError):
        validate_compact(payload, _task(pair))


def test_exact_case_coverage_and_source_constraints():
    with pytest.raises(ValueError, match="exact case"):
        validate_compact(CompactEligibilityOutput(decisions={}), _task())
    constrained = _constrained_output_model(CompactEligibilityOutput, _task())
    valid = {
        "decisions": {
            "k0": {"decision": "neither", "reason": "Not supported", "span_ids": ["S1"]}
        }
    }
    assert constrained.model_validate(valid)
    with pytest.raises(ValueError):
        constrained.model_validate(
            {"decisions": {"k0": {**valid["decisions"]["k0"], "span_ids": ["fake"]}}}
        )


def test_sampling_is_one_paper_stable_across_order_and_retry():
    ids = [f"paper_{i}" for i in range(50)]
    chosen = sampled_verbose_paper("batch", ids)
    assert chosen in ids
    assert chosen == sampled_verbose_paper("batch", list(reversed(ids)) + ids)
    assert sampled_verbose_paper("batch", []) is None
    assert len({sampled_verbose_paper(f"batch_{i}", ids) for i in range(20)}) > 1


@pytest.mark.parametrize("split", [False, True])
def test_compact_primary_and_blind_tiebreak_keep_citations_and_isolation(
    tmp_path, split
):
    candidates = [
        _candidate("bronze:C1", "bronze", None),
        _candidate("miner:uid_1:C2", "miner", "uid_1"),
    ]
    session = _session(tmp_path, candidates)
    session.config = replace(
        session.config,
        eligibility_output_contract="compact",
        adjudication_harness="dspy",
    )
    calls = []

    def run(self, *, task, output_model, stage_key, validator, **kwargs):
        calls.append((stage_key, task))
        if output_model is CompactBlindOutput:
            assert "cases" not in task and "primary_assessments" not in task
            assert "candidate" not in json.dumps(task["source_spans"])
            payload = output_model(
                findings=[
                    {
                        "finding": "Treatment reduced mortality in the trial.",
                        "span_ids": ["S1"],
                    }
                ]
            )
        else:
            if "appellate" in task.get("mode", ""):
                assert task["blind_reconstruction_sha256"]
                assert task["blind_reconstruction"]["findings"]
            decision = (
                "b" if split and task["judge_role"] in {"positive", "tiebreak"} else "a"
            )
            payload = output_model(
                decisions={
                    c["case_ref"]: {
                        "decision": decision,
                        "reason": "Direct trial evidence.",
                        "span_ids": ["S1"],
                    }
                    for c in task["cases"]
                }
            )
        validator(payload)
        return payload

    session._run_dspy_eligibility_stage = MethodType(run, session)
    result = session.run_eligibility_adjudication([_context("case0", candidates)])[0]
    assert result.selected_candidate_id == candidates[1 if split else 0].candidate_id
    assert result.consensus_route == ("tiebreak" if split else "unanimous")
    assert result.cited_span_ids == ["S1"]
    assert all(
        v.output_contract == "compact" and not v.candidate_assessments
        for v in result.primary_votes
    )
    assert bool(result.blind_reconstruction_sha256) == split
    assert len(calls) == (4 if split else 2)


def test_invalid_case_retries_do_not_turn_technical_failure_into_rejection(tmp_path):
    candidate = _candidate("bronze:C1", "bronze", None)
    session = _session(tmp_path, [candidate])
    session.config = replace(
        session.config,
        eligibility_output_contract="compact",
        adjudication_harness="dspy",
    )

    def run(self, *, task, output_model, validator, **kwargs):
        payload = output_model(
            decisions={
                c["case_ref"]: {
                    "decision": "a",
                    "reason": "supported",
                    "span_ids": ["wrong"],
                }
                for c in task["cases"]
            }
        )
        validator(payload)
        return payload

    session._run_dspy_eligibility_stage = MethodType(run, session)
    result = session.run_eligibility_adjudication([_context("case0", [candidate])])[0]
    assert result.resolution_status == "technical_unresolved"
    assert result.consensus_route == "technical_unresolved"


def test_original_candidates_do_not_come_from_judge_atom_text():
    candidate = _candidate("bronze:C1", "bronze", None)
    assert candidate_findings([candidate], [candidate.candidate_id]) == [
        {
            "candidate_ref": candidate.candidate_id,
            "claim_text": candidate.statement,
            "source_span_ids": ["S1"],
            "source_quotes": candidate.source_quotes,
        }
    ]


def test_compact_split_recovery_preserves_successful_sibling_cases(tmp_path):
    candidates = [
        _candidate("bronze:C1", "bronze", None),
        _candidate("bronze:C2", "bronze", None),
    ]
    session = _session(tmp_path, candidates)
    session.config = replace(
        session.config,
        eligibility_output_contract="compact",
        adjudication_harness="dspy",
    )

    def run(self, *, task, output_model, validator, **kwargs):
        # One isolated case is persistently invalid; the other is valid after splitting.
        if any(case["case_ref"] == "k1" for case in task["cases"]):
            raise ValueError("Malformed response for k1")
        payload = output_model(
            decisions={
                case["case_ref"]: {
                    "decision": "a",
                    "reason": "Trial evidence",
                    "span_ids": ["S1"],
                }
                for case in task["cases"]
            }
        )
        validator(payload)
        return payload

    session._run_dspy_eligibility_stage = MethodType(run, session)
    results = session.run_eligibility_adjudication(
        [_context("case0", [candidates[0]]), _context("case1", [candidates[1]])]
    )
    assert results[0].selected_candidate_id == "bronze:C1"
    assert results[0].resolution_status == "decided"
    assert results[1].resolution_status == "technical_unresolved"
