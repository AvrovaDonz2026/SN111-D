from __future__ import annotations

import json
from types import MethodType, SimpleNamespace

import pytest

from validator.agent_v1.comparison_models import CandidatePairEdge, SilverRecord
from validator.agent_v1.equivalence import (
    confirmed_equivalence_groups,
    propose_similarity_groups,
)
from validator.agent_v1.file_agent_workflow import (
    CanonicalAuditOutput,
    CanonicalizationAgentOutput,
    CanonicalQualityChecks,
    ComparisonAgentOutput,
    FileAgentWorkflowError,
)
from validator.agent_v1.orchestrator import (
    MinerArtifactSubmission,
    run_paper_silver_pipeline,
)
from test_validator_agent_v1_file_workflow import (
    _artifact,
    _candidate,
    _session,
    _draft_unit_review,
)


@pytest.fixture(autouse=True)
def offline_models(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("CLAIMS_SILVER_PAIRING_EMBEDDING_MODE", "off")


def edge(left, right, relation="semantic_equivalent"):
    return CandidatePairEdge(
        edge_id=f"{left}-{right}",
        left_candidate_id=left,
        right_candidate_id=right,
        relation=relation,
        confidence=0.99,
    )


def candidates():
    return [
        _candidate(
            "b",
            "bronze",
            None,
            "Treatment reduced mortality.",
            evidence_ids=["EB"],
            source_span_ids=["S1"],
        ),
        _candidate(
            "m1",
            "miner",
            "1",
            "Mortality declined with treatment.",
            evidence_ids=["E1"],
            source_span_ids=["S2"],
        ),
        _candidate(
            "m2",
            "miner",
            "2",
            "Treatment lowered mortality.",
            evidence_ids=["E2"],
            source_span_ids=["S3"],
        ),
    ]


def test_reference_anchored_group_retains_all_miners_and_stable_id():
    kwargs = dict(
        paper_id="paper",
        candidates=candidates(),
        edges=[edge("b", "m1"), edge("b", "m2")],
        explicit_groups=[],
    )
    groups = confirmed_equivalence_groups(**kwargs)
    assert len(groups) == 1
    assert groups[0].member_candidate_ids == ["b", "m1", "m2"]
    assert groups == confirmed_equivalence_groups(
        **{
            **kwargs,
            "candidates": list(reversed(candidates())),
            "edges": list(reversed(kwargs["edges"])),
        }
    )


def test_material_dispute_blocks_bypass_and_unknown_groups_are_ignored():
    groups = confirmed_equivalence_groups(
        paper_id="paper",
        candidates=candidates(),
        edges=[edge("b", "m1"), edge("m1", "m2", "contradiction")],
        explicit_groups=[["b", "m1"], ["m2", "invented"]],
    )
    assert groups == []


def test_no_arbitrary_transitive_closure_but_explicit_miner_only_group_allowed():
    cs = candidates()
    assert (
        confirmed_equivalence_groups(
            paper_id="paper",
            candidates=cs,
            edges=[edge("b", "m1"), edge("m1", "m2")],
            explicit_groups=[],
        )
        == []
    )
    groups = confirmed_equivalence_groups(
        paper_id="paper", candidates=cs, edges=[], explicit_groups=[["m1", "m2"]]
    )
    assert groups[0].member_candidate_ids == ["m1", "m2"]


def test_embedding_proposals_are_complete_link_and_failure_is_optional(monkeypatch):
    import math

    vectors = {
        key: [math.cos(angle), math.sin(angle)]
        for key, angle in [("b", 0), ("m1", 0.3), ("m2", 0.6)]
    }
    groups = propose_similarity_groups(
        candidates(), embedding_provider=lambda _: vectors, threshold=0.9
    )
    assert groups == [["b", "m1"]]  # adjacent similarities cannot chain all three

    def failed(_):
        raise RuntimeError("embedding service unavailable")

    assert propose_similarity_groups(candidates(), embedding_provider=failed) == []


@pytest.mark.parametrize("unknown_source", [False, True])
def test_comparison_persists_groups_and_fresh_session_audits_members(
    tmp_path, unknown_source
):
    cs = candidates()
    if unknown_source:
        cs[2].source_span_ids = ["unknown-paper-span"]
        cs[2].metadata["evidence_records"][0]["source_refs"] = [
            {"span_ids": ["unknown-paper-span"]}
        ]
    session = _session(tmp_path, cs)

    def compare(_self, **kwargs):
        return SimpleNamespace(
            payload=ComparisonAgentOutput(
                submission_reviews=[
                    {
                        "submission_candidate_id": alias,
                        "reference_relations": [
                            {
                                "reference_candidate_id": "c0",
                                "relation": "semantic_equivalent",
                                "confidence": 0.99,
                                "rationale": "Same population, result, scope and scientific qualifiers.",
                            }
                        ],
                    }
                    for alias in ["c1", "c2"]
                ],
                equivalence_groups=[["c0", "c1", "c2"]],
            )
        )

    session._run_stage = MethodType(compare, session)
    session.run_comparison()
    saved = json.loads(
        (session.root / "comparison/equivalence_groups.json").read_text()
    )
    assert saved["groups"][0]["member_candidate_ids"] == ["b", "m1", "m2"]
    # Reload from disk, without relying on the comparison session's memory.
    reloaded = _session(tmp_path, cs)

    def canonical(_self, **kwargs):
        task = kwargs["task"]
        if unknown_source:
            assert task["mandatory_evidence_exclusions"] == [
                {
                    "candidate_id": "c2",
                    "reason": "No linked source span resolves to source text for this paper.",
                }
            ]
        assert task["candidate_admission_status"] == {
            "c0": "pending_source_support_review",
            "c1": "pending_source_support_review",
            "c2": "pending_source_support_review",
        }
        partition = dict(
            units=[
                {
                    "statement": "Treatment reduced mortality.",
                    "candidate_ids": ["c0", "c1"],
                }
            ],
            exclusions=[
                {
                    "candidate_ids": ["c2"],
                    "reason": "Its own linked evidence does not support the claim.",
                }
            ],
        )
        if kwargs["stage_key"] == "canonicalization_draft":
            return SimpleNamespace(payload=CanonicalizationAgentOutput(**partition))
        return SimpleNamespace(
            payload=CanonicalAuditOutput(
                **partition,
                draft_unit_reviews=[_draft_unit_review()],
                quality_checks=CanonicalQualityChecks(
                    duplicate_or_split_attack_checked=True,
                    paper_relevance_checked=True,
                    evidence_support_checked=True,
                    contradiction_checked=True,
                    importance_checked=True,
                ),
            )
        )

    reloaded._run_stage = MethodType(canonical, reloaded)
    record = reloaded.run_canonicalization(
        baseline_record=SilverRecord(silver_record_id="silver", paper_id="paper"),
        decisions=[],
    )
    unit = record.silver_units[0]
    assert unit.equivalent_candidate_ids == ["b", "m1"]
    assert unit.evidence_ids == ["E1", "EB"]
    assert unit.source_span_ids == ["S1", "S2"]
    assert {row["miner_id"] for row in unit.metadata["candidate_provenance"]} == {
        None,
        "1",
    }
    assert unit.adjudication_case_ids == []


@pytest.mark.parametrize("mixed", [False, True])
def test_pipeline_routes_groups_separately_and_keeps_other_cases(tmp_path, mixed):
    from validator.agent_v1.comparison_models import EquivalenceGroup

    class Session:
        config = SimpleNamespace(adjudication_max_workers=1)
        fallback_to_legacy = False

        def run_comparison(self):
            self.confirmed_equivalence_groups = [
                EquivalenceGroup(
                    group_id="eq", member_candidate_ids=["bronze:C01", "miner:one:C01"]
                )
            ]
            return [edge("bronze:C01", "miner:one:C01")]

        def record_comparison_cases(self, cases):
            self.cases = cases

        def run_eligibility_adjudication(self, contexts):
            assert all(
                not {"bronze:C01", "miner:one:C01"}.intersection(c.case.candidate_ids)
                for c in contexts
            )
            assert len(contexts) == int(mixed)
            from validator.agent_v1.eligibility import EligibilityAdjudicationDecision

            return [
                EligibilityAdjudicationDecision(
                    case_id=context.case.case_id,
                    selected_candidate_id=None,
                    rejected_candidate_ids=context.case.candidate_ids,
                    consensus_route="unanimous",
                    rationale="Both judges exclude this unrelated unsupported claim.",
                    primary_votes=[],
                )
                for context in contexts
            ]

        def run_canonicalization(self, *, baseline_record, decisions):
            assert (
                baseline_record.silver_units == []
            )  # pending reference is not auto-included
            return baseline_record

        def finalize(self, **kwargs):
            return {}

    session = Session()
    arts = [
        MinerArtifactSubmission(
            miner_id="one",
            artifact=_artifact("paper", "C01", "Treatment lowered mortality."),
        )
    ]
    if mixed:
        arts.append(
            MinerArtifactSubmission(
                miner_id="two",
                artifact=_artifact("paper", "C02", "A distinct finding."),
            )
        )
    result = run_paper_silver_pipeline(
        paper_id="paper",
        bronze_artifact=_artifact("paper", "C01", "Treatment reduced mortality."),
        miner_artifacts=arts,
        silver_record_id="silver",
        adjudication_passes=[],
        file_agent_workflow=SimpleNamespace(start_session=lambda **kwargs: session),
    )
    assert len(result.equivalence_groups) == 1
    assert len(result.diff_cases) == int(mixed)


def test_pending_exact_restatements_can_all_be_excluded_without_forced_acceptance(
    tmp_path,
):
    cs = candidates()
    # Same statement does not entitle a member to use another member's evidence.
    for candidate in cs:
        candidate.statement = "Treatment reduced mortality."
    session = _session(tmp_path, cs)
    session._write_json(
        "comparison/equivalence_groups.json",
        {
            "schema": "claims_comparison_equivalence_groups_v1",
            "paper_id": "paper",
            "workspace_id": session.workspace_id,
            "groups": [{"group_id": "eq", "member_candidate_ids": ["b", "m1", "m2"]}],
        },
    )

    def stage(_self, **kwargs):
        assert kwargs["task"]["mandatory_same_unit_groups"] == []
        partition = dict(
            units=[],
            exclusions=[
                {
                    "candidate_ids": ["c0", "c1", "c2"],
                    "reason": "None of the supplied passages supports this asserted finding.",
                }
            ],
        )
        if kwargs["stage_key"] == "canonicalization_draft":
            return SimpleNamespace(payload=CanonicalizationAgentOutput(**partition))
        payload = CanonicalAuditOutput(
            **partition,
            draft_unit_reviews=[],
            quality_checks=CanonicalQualityChecks(
                duplicate_or_split_attack_checked=True,
                paper_relevance_checked=True,
                evidence_support_checked=True,
                contradiction_checked=True,
                importance_checked=True,
            ),
        )
        return SimpleNamespace(payload=payload)

    session._run_stage = MethodType(stage, session)
    record = session.run_canonicalization(
        baseline_record=SilverRecord(silver_record_id="silver", paper_id="paper"),
        decisions=[],
    )
    assert record.silver_units == []
    assert len(record.metadata["file_agent_workflow"]["canonical_exclusions"]) == 1


def test_pending_artifact_is_scoped_to_paper_and_workspace(tmp_path):
    session = _session(tmp_path, candidates())
    session._write_json(
        "comparison/equivalence_groups.json",
        {
            "schema": "claims_comparison_equivalence_groups_v1",
            "paper_id": "other-paper",
            "workspace_id": session.workspace_id,
            "groups": [],
        },
    )
    with pytest.raises(FileAgentWorkflowError, match="does not match"):
        session.run_canonicalization(
            baseline_record=SilverRecord(silver_record_id="silver", paper_id="paper"),
            decisions=[],
        )


def test_complete_pipeline_keeps_two_equivalent_miners_and_scores_from_audited_members(
    tmp_path,
):
    stages = []

    class Workflow:
        def start_session(self, **kwargs):
            self.session = _session(
                tmp_path, kwargs["candidates"], workspace_id=kwargs["workspace_id"]
            )
            self.session._run_stage = MethodType(stage, self.session)
            return self.session

    def stage(_self, **kwargs):
        stages.append(kwargs["stage_key"])
        if kwargs["stage_key"] == "comparison":
            return SimpleNamespace(
                payload=ComparisonAgentOutput(
                    submission_reviews=[
                        {
                            "submission_candidate_id": alias,
                            "reference_relations": [
                                {
                                    "reference_candidate_id": "c0",
                                    "relation": "semantic_equivalent",
                                    "confidence": 0.99,
                                    "rationale": "The same qualified scientific result is expressed in both statements.",
                                }
                            ],
                        }
                        for alias in ["c1", "c2"]
                    ],
                    equivalence_groups=[["c0", "c1", "c2"]],
                )
            )
        partition = dict(
            units=[
                {
                    "statement": "Treatment reduced mortality.",
                    "candidate_ids": ["c0", "c1", "c2"],
                }
            ]
        )
        if kwargs["stage_key"] == "canonicalization_draft":
            return SimpleNamespace(payload=CanonicalizationAgentOutput(**partition))
        assert kwargs["stage_key"] == "canonicalization_audit"
        return SimpleNamespace(
            payload=CanonicalAuditOutput(
                **partition,
                draft_unit_reviews=[_draft_unit_review()],
                quality_checks=CanonicalQualityChecks(
                    duplicate_or_split_attack_checked=True,
                    paper_relevance_checked=True,
                    evidence_support_checked=True,
                    contradiction_checked=True,
                    importance_checked=True,
                ),
            )
        )

    def artifact(statement, evidence_id):
        result = _artifact("paper", "C01", statement)
        result["logic"]["claims"][0]["evidence_ids"] = [evidence_id]
        result["evidence"] = {
            "records": [
                {"evidence_id": evidence_id, "source_refs": [{"span_ids": ["S1"]}]}
            ]
        }
        return result

    workflow = Workflow()
    result = run_paper_silver_pipeline(
        paper_id="paper",
        bronze_artifact=artifact("Treatment reduced mortality.", "EB"),
        miner_artifacts=[
            MinerArtifactSubmission(
                miner_id=str(index), artifact=artifact(statement, f"E{index}")
            )
            for index, statement in enumerate(
                ["Treatment lowered mortality.", "Mortality declined with treatment."],
                1,
            )
        ],
        silver_record_id="silver",
        adjudication_passes=[],
        file_agent_workflow=workflow,
    )
    assert stages == ["comparison", "canonicalization_draft", "canonicalization_audit"]
    assert result.diff_cases == []
    assert result.eligibility_adjudication_decisions == []
    assert result.silver_record.silver_units[0].equivalent_candidate_ids == [
        "bronze:C01",
        "miner:1:C01",
        "miner:2:C01",
    ]
    assert [score.coverage for score in result.scores] == [1.0, 1.0]
    assert (
        len(result.silver_record.silver_units[0].metadata["candidate_provenance"]) == 3
    )


def test_disabled_bypass_keeps_comparison_edges_but_clears_saved_groups(tmp_path):
    from dataclasses import replace

    session = _session(tmp_path, candidates()[:2])
    session.config = replace(session.config, equivalence_group_bypass=False)

    def stage(_self, **kwargs):
        return SimpleNamespace(
            payload=ComparisonAgentOutput(
                submission_reviews=[
                    {
                        "submission_candidate_id": "c1",
                        "reference_relations": [
                            {
                                "reference_candidate_id": "c0",
                                "relation": "semantic_equivalent",
                                "confidence": 0.99,
                                "rationale": "Both statements express the same scientific finding with the same scope.",
                            }
                        ],
                    }
                ],
                equivalence_groups=[["c0", "c1"]],
            )
        )

    session._run_stage = MethodType(stage, session)
    assert len(session.run_comparison()) == 1
    assert session.confirmed_equivalence_groups == []
    saved = json.loads(
        (session.root / "comparison/equivalence_groups.json").read_text()
    )
    assert saved["groups"] == []
