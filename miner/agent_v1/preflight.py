from __future__ import annotations

import re
import unicodedata
from typing import Any

from .artifact_models import Artifact, SourceRef
from .artifact_validator import validate_agent_artifact


_HYPHEN_LINEBREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")
_QUALIFIER_RE = re.compile(
    r"\b("
    r"significant(?:ly)?|insignificant|causal(?:ly)?|causation|correlat(?:e|ed|ion|ions)|"
    r"increase(?:d|s)?|decrease(?:d|s)?|improv(?:e|ed|es|ement)|reduc(?:e|ed|es|tion)|"
    r"outperform(?:s|ed)?|superior|best|first|novel|robust(?:ness)?|generaliz(?:e|es|ation)|"
    r"p\s*[<=>]\s*0?\.\d+|n\s*=\s*\d+"
    r")\b",
    re.IGNORECASE,
)


def run_preflight(
    artifact: Artifact,
    *,
    source_payload: dict[str, Any] | None = None,
    require_source_grounding: bool = True,
) -> list[dict[str, Any]]:
    """Deterministic submission gate used before upload / repair feedback.

    Structural checks always run. When ``require_source_grounding`` is true and a
    source payload is available, also enforce span existence, quote-in-span, numeric
    grounding, and a light qualifier-token check.
    """
    issues = validate_agent_artifact(artifact)
    if not require_source_grounding:
        return issues
    if not source_payload:
        issues.append(
            {
                "path": "source_payload",
                "code": "missing_source_payload",
                "message": "source_payload is required for evidence-grounded preflight",
            }
        )
        return issues
    issues.extend(_grounding_issues(artifact, source_payload))
    issues.extend(_qualifier_alignment_issues(artifact, source_payload))
    return _dedupe_issues(issues)


def normalize_quote_text(text: str) -> str:
    """Normalize PDF/OCR text for deterministic quote-in-span checks."""
    value = unicodedata.normalize("NFKC", text or "")
    value = value.replace("\u00ad", "")  # soft hyphen
    value = value.replace("\u2010", "-").replace("\u2011", "-").replace("\u2012", "-")
    value = value.replace("\u2013", "-").replace("\u2014", "-").replace("\u2212", "-")
    value = value.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    value = _HYPHEN_LINEBREAK.sub(r"\1\2", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\n", " ")
    value = _WHITESPACE.sub(" ", value).strip().casefold()
    return value


def quote_in_span(quote: str, span_text: str) -> bool:
    needle = normalize_quote_text(quote)
    haystack = normalize_quote_text(span_text)
    return bool(needle) and needle in haystack


def _grounding_issues(artifact: Artifact, source_payload: dict[str, Any]) -> list[dict[str, Any]]:
    from validator.agent_v1.grounding import run_grounding_checks

    findings = run_grounding_checks(artifact, source_payload)
    issues: list[dict[str, Any]] = []
    spans = {
        str(span.get("span_id")): str(span.get("text") or "")
        for span in source_payload.get("spans", []) or []
        if span.get("span_id")
    }
    for finding in findings:
        code = str((finding.metadata or {}).get("code") or finding.dimension)
        # Re-check quotes with stronger PDF normalization when the validator
        # flagged a quote mismatch; accept if the enhanced matcher passes.
        if code == "quote_not_in_source_span":
            quote = str((finding.metadata or {}).get("quote") or "")
            span_ids = list((finding.metadata or {}).get("span_ids") or [])
            joined = " ".join(spans[span_id] for span_id in span_ids if span_id in spans)
            if quote and joined and quote_in_span(quote, joined):
                continue
        issues.append(
            {
                "path": f"{finding.target_type}.{finding.target_id or ''}.{finding.dimension}".rstrip("."),
                "code": code,
                "message": finding.message,
            }
        )
    # Extra pass: every quote must survive enhanced normalization even if the
    # validator's looser matcher already accepted it (keep symmetry for repair).
    for claim in artifact.logic.claims:
        for ref in claim.sources:
            issues.extend(_enhanced_quote_issues(ref, spans, f"logic.claims.{claim.claim_id}"))
    for record in artifact.evidence.records:
        for ref in record.source_refs:
            issues.extend(_enhanced_quote_issues(ref, spans, f"evidence.records.{record.evidence_id}"))
    for experiment in artifact.logic.experiments:
        for ref in experiment.source_refs:
            issues.extend(_enhanced_quote_issues(ref, spans, f"logic.experiments.{experiment.experiment_id}"))
    return issues


def _enhanced_quote_issues(ref: SourceRef, spans: dict[str, str], path_prefix: str) -> list[dict[str, Any]]:
    quote = (ref.quote or "").strip()
    if not quote or not ref.span_ids:
        return []
    joined = " ".join(spans[span_id] for span_id in ref.span_ids if span_id in spans)
    if not joined:
        return []
    if quote_in_span(quote, joined):
        return []
    return [
        {
            "path": f"{path_prefix}.sources",
            "code": "quote_not_in_source_span",
            "message": "Source quote is not a contiguous normalized substring of the cited span text",
        }
    ]


def _qualifier_alignment_issues(artifact: Artifact, source_payload: dict[str, Any]) -> list[dict[str, Any]]:
    spans = {
        str(span.get("span_id")): str(span.get("text") or "")
        for span in source_payload.get("spans", []) or []
        if span.get("span_id")
    }
    issues: list[dict[str, Any]] = []
    for claim in artifact.logic.claims:
        linked_refs: list[SourceRef] = list(claim.sources)
        for evidence in artifact.evidence.records:
            if evidence.evidence_id in claim.evidence_ids or claim.claim_id in evidence.linked_claim_ids:
                linked_refs.extend(evidence.source_refs)
        connected = " ".join(
            text
            for ref in linked_refs
            for text in ([ref.quote or ""] + [spans[span_id] for span_id in ref.span_ids if span_id in spans])
        )
        if not connected.strip():
            continue
        statement_quals = {match.group(0).casefold() for match in _QUALIFIER_RE.finditer(claim.statement or "")}
        connected_norm = normalize_quote_text(connected)
        missing = sorted(
            token for token in statement_quals if normalize_quote_text(token) not in connected_norm
        )
        if missing:
            issues.append(
                {
                    "path": f"logic.claims.{claim.claim_id}.statement",
                    "code": "claim_qualifier_not_in_source",
                    "message": f"Claim qualifiers absent from connected sources: {', '.join(missing[:8])}",
                }
            )
    return issues


def _dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    unique: list[dict[str, Any]] = []
    for issue in issues:
        key = (str(issue.get("path") or ""), str(issue.get("code") or ""), str(issue.get("message") or ""))
        if key in seen:
            continue
        seen.add(key)
        unique.append(issue)
    return unique
