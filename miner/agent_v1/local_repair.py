from __future__ import annotations

import re
from typing import Any

from .artifact_models import Artifact, SourceRef
from .preflight import _QUALIFIER_RE, normalize_quote_text, quote_in_span


_MIN_QUOTE_CHARS = 24
_MAX_QUOTE_CHARS = 280


def repair_artifact_grounding(
    artifact: Artifact,
    source_payload: dict[str, Any] | None,
) -> tuple[Artifact, list[dict[str, str]]]:
    """Deterministically repair quote/qualifier grounding without calling a model.

    Returns the repaired artifact and a list of repair actions taken.
    """
    if not source_payload:
        return artifact, []
    spans = {
        str(span.get("span_id")): str(span.get("text") or "")
        for span in source_payload.get("spans", []) or []
        if span.get("span_id")
    }
    if not spans:
        return artifact, []

    actions: list[dict[str, str]] = []
    for claim in artifact.logic.claims:
        for ref in claim.sources:
            actions.extend(_repair_source_ref(ref, spans, target=f"claim:{claim.claim_id}"))
        actions.extend(_strip_unsupported_qualifiers(claim, spans))
    for record in artifact.evidence.records:
        for ref in record.source_refs:
            actions.extend(_repair_source_ref(ref, spans, target=f"evidence:{record.evidence_id}"))
    for experiment in artifact.logic.experiments:
        for ref in experiment.source_refs:
            actions.extend(_repair_source_ref(ref, spans, target=f"experiment:{experiment.experiment_id}"))
    return artifact, actions


def _repair_source_ref(ref: SourceRef, spans: dict[str, str], *, target: str) -> list[dict[str, str]]:
    actions: list[dict[str, str]] = []
    quote = (ref.quote or "").strip()
    if not quote:
        return actions

    # 1) Prefer cited spans.
    cited_ids = [span_id for span_id in ref.span_ids if span_id in spans]
    cited_text = " ".join(spans[span_id] for span_id in cited_ids)
    if cited_text and quote_in_span(quote, cited_text):
        return actions

    # 2) Replace paraphrased quote with the best verbatim window from cited spans.
    if cited_text:
        replacement = _best_verbatim_quote(quote, cited_text)
        if replacement and quote_in_span(replacement, cited_text):
            actions.append(
                {
                    "code": "quote_replaced_from_cited_span",
                    "target": target,
                    "message": f"Replaced paraphrased quote with verbatim window ({len(replacement)} chars)",
                }
            )
            ref.quote = replacement
            return actions

    # 3) Search all spans for the original quote or a close verbatim window.
    hit_span_id, hit_text = _find_quote_span(quote, spans)
    if hit_span_id and hit_text:
        if hit_span_id not in ref.span_ids:
            ref.span_ids = [hit_span_id]
            actions.append(
                {
                    "code": "span_retargeted_for_quote",
                    "target": target,
                    "message": f"Retargeted source ref to span `{hit_span_id}`",
                }
            )
        if not quote_in_span(quote, hit_text):
            replacement = _best_verbatim_quote(quote, hit_text) or _fallback_window(hit_text)
            if replacement:
                ref.quote = replacement
                actions.append(
                    {
                        "code": "quote_replaced_from_search",
                        "target": target,
                        "message": f"Installed verbatim quote from `{hit_span_id}`",
                    }
                )
        return actions

    # 4) Last resort: if we have cited spans, install a short verbatim window and keep going.
    if cited_ids:
        window = _fallback_window(spans[cited_ids[0]])
        if window:
            ref.quote = window
            actions.append(
                {
                    "code": "quote_fallback_window",
                    "target": target,
                    "message": f"Installed fallback verbatim window from `{cited_ids[0]}`",
                }
            )
    return actions


def _strip_unsupported_qualifiers(claim, spans: dict[str, str]) -> list[dict[str, str]]:
    linked_text = " ".join(
        text
        for ref in claim.sources
        for text in ([ref.quote or ""] + [spans[span_id] for span_id in ref.span_ids if span_id in spans])
    )
    if not linked_text.strip() or not claim.statement:
        return []
    connected_norm = normalize_quote_text(linked_text)
    missing = [
        match.group(0)
        for match in _QUALIFIER_RE.finditer(claim.statement)
        if normalize_quote_text(match.group(0)) not in connected_norm
    ]
    if not missing:
        return []
    repaired = claim.statement
    for token in missing:
        repaired = re.sub(rf"\b{re.escape(token)}\b", "", repaired, flags=re.IGNORECASE)
    repaired = _WHITESPACE_FIX.sub(" ", repaired).strip(" ,;.")
    repaired = repaired[0].upper() + repaired[1:] if repaired else repaired
    if repaired == claim.statement:
        return []
    claim.statement = repaired
    return [
        {
            "code": "qualifier_stripped",
            "target": f"claim:{claim.claim_id}",
            "message": f"Removed unsupported qualifiers: {', '.join(sorted({t.casefold() for t in missing}))}",
        }
    ]


_WHITESPACE_FIX = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-%\.]*", re.IGNORECASE)


def _find_quote_span(quote: str, spans: dict[str, str]) -> tuple[str, str]:
    for span_id, text in spans.items():
        if quote_in_span(quote, text):
            return span_id, text
    # Rank spans by overlap with quote tokens.
    quote_tokens = set(_TOKEN_RE.findall(normalize_quote_text(quote)))
    if not quote_tokens:
        return "", ""
    best_id = ""
    best_score = 0.0
    best_text = ""
    for span_id, text in spans.items():
        span_tokens = set(_TOKEN_RE.findall(normalize_quote_text(text)))
        if not span_tokens:
            continue
        overlap = len(quote_tokens & span_tokens) / max(1, len(quote_tokens))
        if overlap > best_score:
            best_score = overlap
            best_id = span_id
            best_text = text
    if best_score >= 0.45:
        return best_id, best_text
    return "", ""


def _best_verbatim_quote(quote: str, span_text: str) -> str:
    """Pick a contiguous span window that best covers tokens from the bad quote."""
    norm_span = normalize_quote_text(span_text)
    if not norm_span:
        return ""
    quote_tokens = [tok for tok in _TOKEN_RE.findall(normalize_quote_text(quote)) if len(tok) > 2]
    if not quote_tokens:
        return _fallback_window(span_text)

    # Map normalized char offsets back roughly via sliding windows on original text.
    windows = _candidate_windows(span_text)
    best = ""
    best_score = -1.0
    for window in windows:
        norm_window = normalize_quote_text(window)
        if len(norm_window) < _MIN_QUOTE_CHARS:
            continue
        window_tokens = set(_TOKEN_RE.findall(norm_window))
        hit = sum(1 for tok in quote_tokens if tok in window_tokens)
        score = hit / len(quote_tokens)
        # Prefer shorter windows when scores tie.
        score -= min(0.2, len(norm_window) / 2000.0)
        if score > best_score:
            best_score = score
            best = window.strip()
    if best_score >= 0.35 and best and quote_in_span(best, span_text):
        return best[:_MAX_QUOTE_CHARS].rstrip()
    return ""


def _candidate_windows(text: str) -> list[str]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if not compact:
        return []
    windows: list[str] = []
    # Sentence-ish splits.
    parts = re.split(r"(?<=[.!?])\s+", compact)
    for part in parts:
        part = part.strip()
        if _MIN_QUOTE_CHARS <= len(part) <= _MAX_QUOTE_CHARS:
            windows.append(part)
        elif len(part) > _MAX_QUOTE_CHARS:
            for start in range(0, len(part), _MAX_QUOTE_CHARS // 2):
                chunk = part[start : start + _MAX_QUOTE_CHARS].strip()
                if len(chunk) >= _MIN_QUOTE_CHARS:
                    windows.append(chunk)
    if not windows and len(compact) >= _MIN_QUOTE_CHARS:
        windows.append(compact[:_MAX_QUOTE_CHARS])
    return windows


def _fallback_window(text: str) -> str:
    compact = re.sub(r"\s+", " ", text or "").strip()
    if len(compact) < _MIN_QUOTE_CHARS:
        return compact
    return compact[:_MAX_QUOTE_CHARS].rstrip()


def artifact_repair_rank(artifact: Artifact, issue_count: int) -> tuple[int, int, int]:
    """Higher is better. Prefer non-empty claims, then fewer issues, then more claims."""
    claim_count = len(artifact.logic.claims)
    evidence_count = len(artifact.evidence.records)
    non_empty = 1 if claim_count > 0 else 0
    return (non_empty, -issue_count, claim_count + evidence_count)
