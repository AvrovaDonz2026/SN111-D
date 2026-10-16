"""Small adjudicator wire contracts; verbose policy and models remain available."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .eligibility import (
    ELIGIBILITY_GATES,
    ELIGIBILITY_PROFILE_ID,
    EligibilityAdjudicationVote,
    decide_eligibility_adjudication,
)


class CompactVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["a", "b", "neither"]
    reason: str = Field(min_length=1)
    span_ids: list[str]


class CompactEligibilityOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: dict[str, CompactVerdict]


class CompactBlindFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")
    finding: str = Field(min_length=1)
    span_ids: list[str] = Field(min_length=1)


class CompactBlindOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    findings: list[CompactBlindFinding]
    packet_limitations: list[str] = Field(default_factory=list)


# Serialization guidance only. Judge policy files and system instructions are unchanged.
VERDICT_CONTRACT = {
    "version": "claims_eligibility_compact_v1",
    "instructions": (
        "Apply the governing judge policy and all six gates exactly as before. "
        "Only the serialization changes: return decisions keyed by exact case_ref. "
        "For each case emit only decision, reason, span_ids. decision a/b refers to "
        "the first/second candidate in that case; neither means no candidate qualifies. "
        "Never choose b for a single-candidate case. Put the decisive evidence and "
        "eligibility failure or preference in reason. Cite decisive existing source IDs "
        "in span_ids, including reviewed evidence explaining a rejection. If source spans "
        "are available, cite at least one. Assess every atom and gate, but do not serialize "
        "atom assessments, gates, reviewed_span_ids, or separate candidate rationales. "
        "This contract replaces only the output-field instructions in the policy; "
        "it does not relax its scientific or evidence requirements. Technical failures "
        "are handled by the validator, not expressed as scientific rejection."
    ),
}
BLIND_CONTRACT = {
    "version": "claims_eligibility_compact_blind_v1",
    "instructions": (
        "Preserve the blind reconstruction policy. Return findings as a list of objects "
        "with only finding and span_ids, plus packet_limitations for evidence limitations. "
        "Each finding must contain complete canonical wording and all material qualifications. "
        "span_ids is the union of its assertion, original-support, support-path and salience "
        "anchors. Do not serialize separate anchor categories or repeat qualifications in "
        "another field. Reconstruct every qualifying finding without ranking or capping. "
        "Only the output representation changes. No candidates or primary votes are available."
    ),
}


def validate_compact(payload: CompactEligibilityOutput, task: dict[str, Any]) -> None:
    cases = {case["case_ref"]: case for case in task["cases"]}
    if set(payload.decisions) != set(cases):
        raise ValueError("Compact eligibility must cover the exact case set.")
    known = set(task.get("source_spans") or {})
    for ref, verdict in payload.decisions.items():
        if not verdict.reason.strip():
            raise ValueError("Compact eligibility reason must not be blank.")
        if verdict.decision == "b" and len(cases[ref]["candidates"]) != 2:
            raise ValueError("A single-candidate case cannot select b.")
        if set(verdict.span_ids) - known:
            raise ValueError("Compact eligibility cited unknown source spans.")
        if known and not verdict.span_ids:
            raise ValueError(
                "Compact eligibility must cite decisive or reviewed evidence."
            )
        if not known and verdict.decision != "neither":
            raise ValueError("A candidate cannot pass without source evidence.")


def validate_blind(payload: CompactBlindOutput, task: dict[str, Any]) -> None:
    known = set(task.get("source_spans") or {})
    for finding in payload.findings:
        if not finding.finding.strip() or set(finding.span_ids) - known:
            raise ValueError(
                "Compact blind reconstruction has invalid wording or source spans."
            )


def sampled_verbose_paper(batch_id: str, paper_ids: list[str]) -> str | None:
    """Uniform hash sampling, stable across ordering, restarts and validator runs."""
    return min(
        set(paper_ids),
        key=lambda p: hashlib.sha256(
            json.dumps(["eligibility_verbose_sample_v1", batch_id, p]).encode()
        ).digest(),
        default=None,
    )


def candidate_findings(candidates, candidate_ids: list[str]) -> list[dict[str, Any]]:
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        {
            "candidate_ref": ref,
            "claim_text": by_id[ref].statement,
            "source_span_ids": list(by_id[ref].source_span_ids),
            "source_quotes": list(by_id[ref].source_quotes),
        }
        for ref in candidate_ids
        if ref in by_id
    ]


def _run(session, *, task, output_model, stage_key, model, skill_name, validator):
    """Preserve valid sibling cases; isolate repeated failures as operational errors."""
    from .file_agent_workflow import _skill_path, _eligibility_split_task

    errors = []
    for attempt in range(2):
        key = stage_key if attempt == 0 else stage_key + "_retry"
        current = (
            task
            if not errors
            else {**task, "operational_retry": {"previous_error": errors[-1]}}
        )
        try:
            args = dict(
                stage_key=key,
                stage_label=stage_key,
                model=model,
                task=current,
                output_model=output_model,
                skill_path=_skill_path(skill_name),
            )
            if session.config.adjudication_harness == "dspy" or attempt:
                payload = session._run_dspy_eligibility_stage(
                    **args, validator=lambda p: validator(p, task)
                )
            else:
                payload = session._run_stage(**args).payload
                validator(payload, task)
            return payload, {}
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    cases = task.get("cases") or []
    if output_model is not CompactEligibilityOutput:
        raise ValueError("Blind reconstruction failed: " + errors[-1])
    if len(cases) <= 1:
        failures = {case["case_ref"]: errors[-1] for case in cases}
        session._record_manifest_stage(
            {
                "stage_key": stage_key,
                "status": "technical_unresolved",
                "output_contract": "compact",
                "errors": failures,
            }
        )
        return CompactEligibilityOutput(decisions={}), failures
    decisions, failures = {}, {}
    for index, subset in enumerate(
        (cases[: len(cases) // 2], cases[len(cases) // 2 :])
    ):
        subtask = _eligibility_split_task(
            task, cases=subset, parent_case_count=len(cases)
        )
        result, failed = _run(
            session,
            task=subtask,
            output_model=output_model,
            stage_key=f"{stage_key}_s{index}",
            model=model,
            skill_name=skill_name,
            validator=validator,
        )
        decisions.update(result.decisions)
        failures.update(failed)
    return CompactEligibilityOutput(decisions=decisions), failures


def run_compact_batch(session, contexts, *, batch_index):
    from .file_agent_workflow import (
        build_eligibility_adjudication_task,
        _eligibility_split_task,
    )

    suffix = "" if batch_index == 0 else f"_b{batch_index:03d}"
    source_map = (
        session.eligibility_source_context_by_span_id
        if session.eligibility_source_context_by_span_id is not None
        else session.source_context_by_span_id
    )
    task, aliases, ids = build_eligibility_adjudication_task(
        contexts,
        paper_context=session.paper_context,
        source_context_by_span_id=source_map,
    )
    # Remove verbose serialization requirements; preserve substantive judge requirements.
    task["requirements"] = {
        k: v
        for k, v in task["requirements"].items()
        if k
        not in {
            "failed_candidate_must_list_reviewed_source_spans",
        }
    }
    task["output_contract"] = VERDICT_CONTRACT

    def run(role, model):
        return _run(
            session,
            task={**task, "judge_role": role},
            output_model=CompactEligibilityOutput,
            stage_key=f"eligibility_adjudication_{role}{suffix}",
            model=model,
            skill_name=f"claims-silver-eligibility-selection-{role}",
            validator=validate_compact,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        negative = pool.submit(
            run, "negative", session.config.adjudication_negative_model
        )
        positive = pool.submit(
            run, "positive", session.config.adjudication_positive_model
        )
        primary = {"negative": negative.result(), "positive": positive.result()}

    def vote(ref, role, result, model):
        output, errors = result
        verdict = output.decisions.get(ref)
        if verdict is None:
            return EligibilityAdjudicationVote(
                case_id=next(k for k, v in aliases.items() if v == ref),
                judge_role=role,
                candidate_assessments=[],
                rationale=errors[ref],
                model=model,
                technical_unresolved=True,
                technical_error=errors[ref],
                output_contract="compact",
            )
        candidate_refs = list(ids[ref])
        selected = (
            None
            if verdict.decision == "neither"
            else ids[ref][candidate_refs[0 if verdict.decision == "a" else 1]]
        )
        return EligibilityAdjudicationVote(
            case_id=next(k for k, v in aliases.items() if v == ref),
            judge_role=role,
            selected_candidate_id=selected,
            candidate_assessments=[],
            rationale=verdict.reason,
            model=model,
            cited_span_ids=verdict.span_ids,
            output_contract="compact",
        )

    negatives = {
        r: vote(
            r,
            "negative",
            primary["negative"],
            session.config.adjudication_negative_model,
        )
        for r in ids
    }
    positives = {
        r: vote(
            r,
            "positive",
            primary["positive"],
            session.config.adjudication_positive_model,
        )
        for r in ids
    }
    disputed = [
        r
        for r in ids
        if not negatives[r].technical_unresolved
        and not positives[r].technical_unresolved
        and negatives[r].selected_candidate_id != positives[r].selected_candidate_id
    ]
    tiebreaks, blind_hash = {}, ""
    if disputed:
        split = _eligibility_split_task(
            task,
            cases=[c for c in task["cases"] if c["case_ref"] in disputed],
            parent_case_count=len(task["cases"]),
        )
        blind_task = dict(
            mode="eligibility_blind_reconstruction",
            eligibility_profile_id=ELIGIBILITY_PROFILE_ID,
            paper=session.paper_context,
            source_spans=split["source_spans"],
            hard_gates=list(ELIGIBILITY_GATES),
            output_contract=BLIND_CONTRACT,
        )
        try:
            blind, _ = _run(
                session,
                task=blind_task,
                output_model=CompactBlindOutput,
                stage_key=f"eligibility_adjudication_blind_reconstruction{suffix}",
                model=session.config.adjudication_tiebreak_model,
                skill_name="claims-silver-eligibility-blind-reconstruction",
                validator=validate_blind,
            )
            locked = blind.model_dump(mode="json")
            blind_hash = hashlib.sha256(
                json.dumps(
                    locked, sort_keys=True, ensure_ascii=False, separators=(",", ":")
                ).encode()
            ).hexdigest()
            appellate = {
                **split,
                "mode": "eligibility_appellate_comparison",
                "judge_role": "tiebreak",
                "blind_reconstruction": locked,
                "blind_reconstruction_sha256": blind_hash,
                "primary_assessments": [
                    {
                        "case_ref": r,
                        "negative": primary["negative"][0]
                        .decisions[r]
                        .model_dump(mode="json"),
                        "positive": primary["positive"][0]
                        .decisions[r]
                        .model_dump(mode="json"),
                    }
                    for r in disputed
                ],
                "requirements": {
                    **task["requirements"],
                    "do_not_modify_blind_reconstruction": True,
                    "resolve_each_primary_disagreement": True,
                    "return_only_disputed_cases": True,
                },
            }
            result = _run(
                session,
                task=appellate,
                output_model=CompactEligibilityOutput,
                stage_key=f"eligibility_adjudication_tiebreak{suffix}",
                model=session.config.adjudication_tiebreak_model,
                skill_name="claims-silver-eligibility-selection-tiebreak",
                validator=validate_compact,
            )
        except Exception as exc:
            result = (
                CompactEligibilityOutput(decisions={}),
                {r: f"{type(exc).__name__}: {exc}" for r in disputed},
            )
        tiebreaks = {
            r: vote(r, "tiebreak", result, session.config.adjudication_tiebreak_model)
            for r in disputed
        }
    decisions = [
        decide_eligibility_adjudication(
            case_id=context.case.case_id,
            candidate_ids=[c.candidate_id for c in context.candidates],
            negative_vote=negatives[aliases[context.case.case_id]],
            positive_vote=positives[aliases[context.case.case_id]],
            tiebreak_vote=tiebreaks.get(aliases[context.case.case_id]),
            blind_reconstruction_sha256=blind_hash
            if aliases[context.case.case_id] in disputed
            else "",
        )
        for context in contexts
    ]
    session._write_json(
        f"eligibility_adjudication/batches/batch_{batch_index:03d}.json",
        {
            "output_contract": "compact",
            "decisions": [d.model_dump(mode="json") for d in decisions],
        },
    )
    return decisions
