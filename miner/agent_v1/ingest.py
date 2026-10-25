from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .artifact_models import Paper


class InputSpan(BaseModel):
    span_id: str
    paper_id: str
    section_name: str = ""
    section_type: str = "OTHER"
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    text: str
    span_type: str = "text"
    metadata: dict[str, Any] = Field(default_factory=dict)


class InputDocument(BaseModel):
    paper: Paper
    spans: list[InputSpan] = Field(default_factory=list)
    source_path: str | None = None
    source_type: str = "unknown"
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


PDF_READERS = ("pdf-inspector", "pypdf", "grobid")
SOURCE_PAYLOAD_SCHEMA_VERSION = "agent_v1_source_payload_v1"
SOURCE_PAYLOAD_POLICY_VERSION = "full-source-v1"


def ingest_pdf(
    pdf_path: Path,
    *,
    max_chars: int | None,
    reader: str = "pdf-inspector",
    grobid_url: str = "http://localhost:8070/",
    grobid_cache_dir: Path | None = None,
    grobid_timeout_s: int = 120,
    grobid_retries: int = 3,
    grobid_retry_wait_s: int = 2,
) -> InputDocument:
    paper_id = pdf_path.stem
    normalized_reader = _normalize_pdf_reader(reader)
    if normalized_reader == "pdf-inspector":
        return _document_from_pdf_inspector(pdf_path, paper_id=paper_id, max_chars=max_chars)
    if normalized_reader == "grobid":
        return _document_from_grobid(
            pdf_path,
            max_chars=max_chars,
            grobid_url=grobid_url,
            grobid_cache_dir=grobid_cache_dir,
            grobid_timeout_s=grobid_timeout_s,
            grobid_retries=grobid_retries,
            grobid_retry_wait_s=grobid_retry_wait_s,
        )
    spans = _spans_from_pypdf(pdf_path, paper_id=paper_id, max_chars=max_chars)
    return InputDocument(
        paper=Paper(paper_id=paper_id, title=paper_id),
        spans=spans,
        source_path=str(pdf_path),
        source_type="pdf",
        raw_metadata={"pdf_reader": normalized_reader},
    )


def ingest_artifact_json(path: Path, *, max_chars: int | None) -> InputDocument:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paper_raw = payload.get("paper", {}) if isinstance(payload, dict) else {}
    paper_id = str(paper_raw.get("paper_id") or path.stem)
    paper = Paper(
        paper_id=paper_id,
        title=str(paper_raw.get("title") or paper_id),
        authors=[str(item) for item in paper_raw.get("authors", []) or []],
        year=paper_raw.get("year"),
        venue=paper_raw.get("journal") or paper_raw.get("venue"),
        doi=paper_raw.get("doi"),
    )
    spans = []
    for index, raw in enumerate(payload.get("spans", []) or [], start=1):
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        spans.append(
            InputSpan(
                span_id=str(raw.get("span_id") or f"{paper_id}-span-{index:04d}"),
                paper_id=paper_id,
                section_name=str(raw.get("section_name") or ""),
                section_type=str(raw.get("section_type") or "OTHER"),
                page=raw.get("page"),
                text=text,
                span_type=str(raw.get("span_type") or "text"),
            )
        )
    return InputDocument(
        paper=paper,
        spans=_truncate_spans(spans, max_chars=max_chars),
        source_path=str(path),
        source_type="artifact_json",
        raw_metadata={"input_keys": sorted(payload.keys()) if isinstance(payload, dict) else []},
    )


def ingest_text(text_path: Path, *, max_chars: int | None) -> InputDocument:
    paper_id = text_path.stem
    text = text_path.read_text(encoding="utf-8")
    chunks = _chunk_text(text, max_chars=3500)
    spans = [
        InputSpan(
            span_id=f"{paper_id}-span-{index:04d}",
            paper_id=paper_id,
            section_name=f"Chunk {index}",
            section_type="OTHER",
            text=chunk,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]
    return InputDocument(
        paper=Paper(paper_id=paper_id, title=paper_id),
        spans=_truncate_spans(spans, max_chars=max_chars),
        source_path=str(text_path),
        source_type="text",
    )


def document_source_payload(document: InputDocument, *, max_chars: int | None) -> dict[str, Any]:
    annotated = [_annotate_section(span) for span in document.spans]
    spans = _truncate_spans(annotated, max_chars=max_chars)
    source_text = "\n\n".join(span.text for span in spans)
    complete_text = "\n\n".join(span.text for span in annotated)
    truncated = len(spans) != len(annotated) or source_text != complete_text
    section_index = build_section_index(spans)
    extraction_quality = score_extraction_quality(annotated, document.raw_metadata)
    return {
        "schema_version": SOURCE_PAYLOAD_SCHEMA_VERSION,
        "policy_version": SOURCE_PAYLOAD_POLICY_VERSION,
        "paper": document.paper.model_dump(mode="json"),
        "source_type": document.source_type,
        "source_path": document.source_path,
        "source_metadata": document.raw_metadata,
        "spans": [span.model_dump(mode="json") for span in spans],
        "section_index": section_index,
        "extraction_quality": extraction_quality,
        "source_integrity": {
            "schema": "claims_source_payload_integrity_v1",
            "truncated": truncated,
            "span_count": len(spans),
            "character_count": sum(len(span.text) for span in spans),
            "page_count": len({span.page for span in spans if span.page is not None}),
            "content_sha256": hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
            "complete_span_count": len(annotated),
            "complete_character_count": sum(len(span.text) for span in annotated),
        },
    }


_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("ABSTRACT", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?abstract\b", re.I | re.M)),
    ("INTRODUCTION", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?introduction\b", re.I | re.M)),
    ("METHODS", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?(?:methods?|materials?\s+and\s+methods|methodology)\b", re.I | re.M)),
    ("RESULTS", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?results?\b", re.I | re.M)),
    ("DISCUSSION", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?discussion\b", re.I | re.M)),
    ("LIMITATIONS", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?limitations?\b", re.I | re.M)),
    ("CONCLUSION", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?conclusions?\b", re.I | re.M)),
    ("APPENDIX", re.compile(r"^\s{0,3}(?:#+\s*)?(?:\d+(?:\.\d+)*\s+)?appendix\b", re.I | re.M)),
    ("TABLE", re.compile(r"^\s{0,3}(?:#+\s*)?table\s+\d+", re.I | re.M)),
    ("FIGURE", re.compile(r"^\s{0,3}(?:#+\s*)?figure\s+\d+", re.I | re.M)),
)


def _annotate_section(span: InputSpan) -> InputSpan:
    """Infer coarse section labels for page-level readers that lack TEI structure."""
    if span.section_type and span.section_type not in {"OTHER", "PAGE"}:
        return span
    inferred = _infer_section_type(span.text)
    if not inferred:
        return span
    section_name = span.section_name or inferred.title()
    metadata = dict(span.metadata)
    metadata["inferred_section_type"] = inferred
    return span.model_copy(update={"section_type": inferred, "section_name": section_name, "metadata": metadata})


def _infer_section_type(text: str) -> str:
    head = "\n".join((text or "").splitlines()[:12])
    for section_type, pattern in _SECTION_PATTERNS:
        if pattern.search(head):
            return section_type
    return ""


def build_section_index(spans: list[InputSpan]) -> dict[str, Any]:
    by_type: dict[str, list[str]] = {}
    tables: list[str] = []
    figures: list[str] = []
    for span in spans:
        section_type = (span.section_type or "OTHER").upper()
        by_type.setdefault(section_type, []).append(span.span_id)
        lowered = span.text[:240].casefold()
        if section_type == "TABLE" or lowered.startswith("table "):
            tables.append(span.span_id)
        if section_type == "FIGURE" or lowered.startswith("figure "):
            figures.append(span.span_id)
    return {
        "schema": "claims_section_index_v1",
        "by_section_type": by_type,
        "table_span_ids": tables,
        "figure_caption_span_ids": figures,
        "section_types_present": sorted(by_type),
    }


def score_extraction_quality(spans: list[InputSpan], raw_metadata: dict[str, Any]) -> dict[str, Any]:
    character_count = sum(len(span.text) for span in spans)
    section_types = {(span.section_type or "OTHER").upper() for span in spans}
    useful_sections = section_types & {
        "ABSTRACT",
        "INTRODUCTION",
        "METHODS",
        "RESULTS",
        "DISCUSSION",
        "LIMITATIONS",
        "CONCLUSION",
        "TABLE",
        "FIGURE",
    }
    pages_needing_ocr = list(raw_metadata.get("pages_needing_ocr") or [])
    empty = character_count == 0 or not spans
    score = 0.0
    if not empty:
        score += 0.35
    if character_count >= 2000:
        score += 0.2
    if len(useful_sections) >= 3:
        score += 0.25
    elif len(useful_sections) >= 1:
        score += 0.1
    if raw_metadata.get("pdf_reader") == "grobid":
        score += 0.1
    if pages_needing_ocr:
        score -= min(0.25, 0.05 * len(pages_needing_ocr))
    if bool(raw_metadata.get("is_complex")):
        score -= 0.05
    score = max(0.0, min(1.0, round(score, 3)))
    return {
        "schema": "claims_extraction_quality_v1",
        "score": score,
        "empty": empty,
        "character_count": character_count,
        "span_count": len(spans),
        "useful_section_count": len(useful_sections),
        "useful_sections": sorted(useful_sections),
        "pages_needing_ocr": pages_needing_ocr,
        "recommend_reader_fallback": empty or score < 0.35 or bool(pages_needing_ocr),
    }


def apply_paper_metadata_override(
    document: InputDocument,
    *,
    paper_id: str = "",
    title: str = "",
    source_url: str = "",
    source_sha256: str = "",
) -> InputDocument:
    clean_paper_id = str(paper_id or "").strip()
    clean_title = str(title or "").strip()
    previous_paper_id = document.paper.paper_id
    if clean_paper_id:
        document.paper.paper_id = clean_paper_id
    if clean_title:
        document.paper.title = clean_title
    if clean_paper_id and clean_paper_id != previous_paper_id:
        for span in document.spans:
            span.paper_id = clean_paper_id
            if span.span_id.startswith(f"{previous_paper_id}-"):
                span.span_id = f"{clean_paper_id}{span.span_id[len(previous_paper_id):]}"
            reader_span_id = str(span.metadata.get("reader_span_id") or "")
            if reader_span_id.startswith(f"{previous_paper_id}-"):
                span.metadata["reader_span_id"] = f"{clean_paper_id}{reader_span_id[len(previous_paper_id):]}"
    override = {
        key: value
        for key, value in {
            "paper_id": clean_paper_id,
            "title": clean_title,
            "source_url": str(source_url or "").strip(),
            "source_sha256": str(source_sha256 or "").strip(),
        }.items()
        if value
    }
    if override:
        document.raw_metadata["paper_metadata_override"] = override
    return document


def _normalize_pdf_reader(reader: str) -> str:
    normalized = str(reader or "").strip().lower().replace("_", "-")
    aliases = {
        "pdfinspector": "pdf-inspector",
        "pdf-inspector": "pdf-inspector",
        "pypdf": "pypdf",
        "py-pdf": "pypdf",
        "grobid": "grobid",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise ValueError(f"Unsupported agent_v1 PDF reader: {reader}. Expected one of: {', '.join(PDF_READERS)}")


def _document_from_pdf_inspector(pdf_path: Path, *, paper_id: str, max_chars: int | None) -> InputDocument:
    try:
        import pdf_inspector  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local install
        raise RuntimeError("pdf-inspector is required for agent_v1 PDF ingestion. Install with `pip install pdf-inspector`.") from exc

    result = pdf_inspector.extract_pages_markdown(str(pdf_path))
    spans: list[InputSpan] = []
    for page in getattr(result, "pages", []) or []:
        page_index = int(getattr(page, "page", 0))
        markdown = str(getattr(page, "markdown", "") or "").strip()
        if not markdown:
            continue
        reader_span_id = f"{paper_id}-p{page_index + 1:03d}-markdown"
        spans.append(
            InputSpan(
                span_id=f"{paper_id}-span-{len(spans) + 1:04d}",
                paper_id=paper_id,
                section_name=f"Page {page_index + 1}",
                section_type="PAGE",
                page=page_index + 1,
                text=markdown,
                span_type="text",
                metadata={"reader_span_id": reader_span_id, "pdf_reader": "pdf-inspector"},
            )
        )
    return InputDocument(
        paper=Paper(paper_id=paper_id, title=paper_id),
        spans=_truncate_spans(spans, max_chars=max_chars),
        source_path=str(pdf_path),
        source_type="pdf",
        raw_metadata={
            "pdf_reader": "pdf-inspector",
            "pages_with_tables": list(getattr(result, "pages_with_tables", []) or []),
            "pages_with_columns": list(getattr(result, "pages_with_columns", []) or []),
            "pages_needing_ocr": list(getattr(result, "pages_needing_ocr", []) or []),
            "is_complex": bool(getattr(result, "is_complex", False)),
        },
    )


def _document_from_grobid(
    pdf_path: Path,
    *,
    max_chars: int | None,
    grobid_url: str,
    grobid_cache_dir: Path | None,
    grobid_timeout_s: int,
    grobid_retries: int,
    grobid_retry_wait_s: int,
) -> InputDocument:
    from miner.v0.grobid_client import GrobidClient
    from miner.v0.tei_parser import TEIParser

    cache_dir = grobid_cache_dir or Path(__file__).resolve().parent / ".cache" / "grobid"
    client = GrobidClient(
        grobid_url,
        cache_dir,
        timeout_s=grobid_timeout_s,
        retries=grobid_retries,
        retry_wait_s=grobid_retry_wait_s,
    )
    tei_string = client.process_pdf(pdf_path, use_cache=True)
    parser = TEIParser()
    v0_paper = parser.parse_paper(tei_string, pdf_path)
    v0_spans = parser.extract_spans(tei_string, v0_paper.paper_id)
    paper = Paper(
        paper_id=v0_paper.paper_id,
        title=v0_paper.title or v0_paper.paper_id,
        authors=list(v0_paper.authors or []),
        year=v0_paper.year,
        venue=v0_paper.journal,
        doi=v0_paper.doi,
    )
    spans = [
        InputSpan(
            span_id=span.span_id,
            paper_id=span.paper_id,
            section_name=span.section_name or "",
            section_type=span.section_type or "OTHER",
            page=span.page,
            char_start=span.char_start,
            char_end=span.char_end,
            text=span.text,
            span_type=span.span_type,
        )
        for span in v0_spans
        if span.text
    ]
    return InputDocument(
        paper=paper,
        spans=_truncate_spans(spans, max_chars=max_chars),
        source_path=str(pdf_path),
        source_type="pdf",
        raw_metadata={"pdf_reader": "grobid"},
    )


def _spans_from_pypdf(pdf_path: Path, *, paper_id: str, max_chars: int | None) -> list[InputSpan]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pypdf is required for agent_v1 PDF ingestion.") from exc
    reader = PdfReader(str(pdf_path))
    spans: list[InputSpan] = []
    for page_index, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", page_text) if block.strip()]
        for para_index, paragraph in enumerate(paragraphs, start=1):
            reader_span_id = f"{paper_id}-p{page_index:03d}-{para_index:03d}"
            spans.append(
                InputSpan(
                    span_id=f"{paper_id}-span-{len(spans) + 1:04d}",
                    paper_id=paper_id,
                    section_name=f"Page {page_index}",
                    section_type="PAGE",
                    page=page_index,
                    text=paragraph,
                    metadata={"reader_span_id": reader_span_id, "pdf_reader": "pypdf"},
                )
            )
    return _truncate_spans(spans, max_chars=max_chars)


def _truncate_spans(spans: list[InputSpan], *, max_chars: int | None) -> list[InputSpan]:
    if max_chars is None:
        return list(spans)
    kept: list[InputSpan] = []
    total = 0
    for span in spans:
        if total >= max_chars:
            break
        remaining = max_chars - total
        text = span.text
        if len(text) > remaining:
            text = text[:remaining].rstrip()
        if text:
            kept.append(span.model_copy(update={"text": text}))
            total += len(text)
    return kept


def _chunk_text(text: str, *, max_chars: int) -> list[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        if current and current_len + len(paragraph) + 2 > max_chars:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(paragraph)
        current_len += len(paragraph) + 2
    if current:
        chunks.append("\n\n".join(current))
    return chunks or [text[:max_chars]]
