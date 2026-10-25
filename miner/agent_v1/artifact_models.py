from __future__ import annotations

from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    source_id: str
    source_type: str
    path: str | None = None
    span_ids: list[str] = Field(default_factory=list)
    quote: str | None = None
    role: Literal["input", "result", "method", "interpretation", "metadata"] = "input"


class Paper(BaseModel):
    paper_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    domain: str | None = None
    keywords: list[str] = Field(default_factory=list)
    abstract: str = ""
    claims_summary: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_id: str
    statement: str
    conditions: str
    status: str
    falsification_criteria: str
    proof: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    sources: list[SourceRef] = Field(default_factory=list)
    source_claim_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Concept(BaseModel):
    concept_id: str
    label: str
    definition: str
    source_refs: list[SourceRef] = Field(default_factory=list)


class Experiment(BaseModel):
    experiment_id: str
    title: str
    verifies: list[str] = Field(default_factory=list)
    setup: str
    procedure: str
    expected_outcome: str
    evidence_ids: list[str] = Field(default_factory=list)
    run: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    evidence_id: str
    title: str
    role: str
    summary: str
    evidence_method: str = ""
    outcome_type: str | None = None
    presentation_type: str | None = None
    source_refs: list[SourceRef] = Field(default_factory=list)
    linked_claim_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TraceNode(BaseModel):
    node_id: str
    node_type: str
    support_level: Literal["explicit", "inferred"] = "inferred"
    summary: str
    source_refs: list[SourceRef] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    children: list["TraceNode"] = Field(default_factory=list)


class Logic(BaseModel):
    problem_observations: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    key_insight: str = ""
    assumptions: list[str] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    concepts: list[Concept] = Field(default_factory=list)
    experiments: list[Experiment] = Field(default_factory=list)
    related_work: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class EvidenceLayer(BaseModel):
    records: list[EvidenceRecord] = Field(default_factory=list)
    ledger_notes: list[str] = Field(default_factory=list)


class SrcLayer(BaseModel):
    environment: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class Artifact(BaseModel):
    ara_version: str = "1.0"
    paper: Paper
    logic: Logic
    evidence: EvidenceLayer
    trace: TraceNode
    src: SrcLayer
    metadata: dict[str, Any] = Field(default_factory=dict)


TraceNode.model_rebuild()


def unknown_field_diagnostics(
    value: Any,
    model_type: type[BaseModel] = Artifact,
) -> list[dict[str, Any]]:
    """Report keys that would be discarded by Pydantic's default extra policy.

    The artifact models intentionally keep Pydantic's backwards-compatible
    ``extra='ignore'`` behavior.  This helper makes that behavior observable
    without changing model construction.  Arbitrary ``dict[str, Any]`` fields
    such as ``metadata`` are treated as extension points and are not inspected.
    """

    diagnostics: list[dict[str, Any]] = []

    def visit(current: Any, current_model: type[BaseModel], path: str) -> None:
        if not isinstance(current, dict):
            return

        fields = current_model.model_fields
        for key in current:
            if key not in fields:
                diagnostics.append(
                    {
                        "path": f"{path}.{key}",
                        "code": "unknown_field",
                        "field": key,
                        "model": current_model.__name__,
                        "message": f"Unknown field '{key}' would be ignored by {current_model.__name__}.",
                    }
                )

        for field_name, field_info in fields.items():
            if field_name not in current:
                continue
            child = current[field_name]
            child_model = _nested_model_type(field_info.annotation)
            if child_model is not None:
                visit(child, child_model, f"{path}.{field_name}")
                continue
            item_model = _nested_list_item_model(field_info.annotation)
            if item_model is not None and isinstance(child, list):
                for index, item in enumerate(child):
                    visit(item, item_model, f"{path}.{field_name}[{index}]")

    visit(value, model_type, "$")
    return diagnostics


def _nested_model_type(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        for argument in get_args(annotation):
            nested = _nested_model_type(argument)
            if nested is not None:
                return nested
    return None


def _nested_list_item_model(annotation: Any) -> type[BaseModel] | None:
    if get_origin(annotation) is list:
        arguments = get_args(annotation)
        if arguments:
            return _nested_model_type(arguments[0])
    return None
