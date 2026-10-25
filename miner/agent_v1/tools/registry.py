from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..artifact import materialize_agent_artifact
from ..preflight import normalize_quote_text, run_preflight
from ..skillpack import SkillPack


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    func: Callable[..., str]


class AgentToolbox:
    def __init__(self, *, run_dir: Path, skill_pack: SkillPack) -> None:
        self.run_dir = run_dir.resolve()
        self.skill_pack = skill_pack

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec("read_source_payload", "Read the full source payload JSON for this run.", self.read_source_payload),
            ToolSpec(
                "read_source_span",
                "Read one source span by ID with page, neighbors, and normalized text.",
                self.read_source_span,
            ),
            ToolSpec("read_run_file", "Read a UTF-8 file inside the current run directory.", self.read_run_file),
            ToolSpec("write_run_file", "Write a UTF-8 file inside the current run directory.", self.write_run_file),
            ToolSpec("list_run_files", "List files inside the current run directory.", self.list_run_files),
            ToolSpec("search_source_text", "Search source span text for a case-insensitive query.", self.search_source_text),
            ToolSpec("read_section_index", "Read the section/table/figure index for this paper.", self.read_section_index),
            ToolSpec("read_output_schema", "Read the JSON Schema for the required agent_output.json payload.", self.read_output_schema),
            ToolSpec("read_skill_resource", "Read a mounted skill resource by relative path.", self.read_skill_resource),
            ToolSpec("validate_agent_artifact", "Validate candidate agent JSON and return validation issues.", self.validate_agent_artifact),
            ToolSpec("submit_agent_artifact", "Validate and write the final agent_output.json file.", self.submit_agent_artifact),
        ]

    def read_source_payload(self) -> str:
        return self._read_file("source_payload.json")

    def read_run_file(self, path: str) -> str:
        return self._read_file(path)

    def write_run_file(self, path: str, content: str) -> str:
        target = self._safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"wrote {target.relative_to(self.run_dir).as_posix()}"

    def list_run_files(self) -> str:
        files = sorted(path.relative_to(self.run_dir).as_posix() for path in self.run_dir.rglob("*") if path.is_file())
        return json.dumps(files, indent=2)

    def search_source_text(self, query: str, limit: int = 20) -> str:
        payload = json.loads(self.read_source_payload())
        needle = query.casefold()
        matches = []
        for span in payload.get("spans", []) or []:
            text = str(span.get("text") or "")
            if needle and needle in text.casefold():
                matches.append(
                    {
                        "span_id": span.get("span_id"),
                        "section_name": span.get("section_name"),
                        "section_type": span.get("section_type"),
                        "page": span.get("page"),
                        "text": text[:1200],
                    }
                )
            if len(matches) >= limit:
                break
        return json.dumps(matches, indent=2, ensure_ascii=False)

    def read_source_span(self, span_id: str, neighbors: int = 1) -> str:
        payload = json.loads(self.read_source_payload())
        spans = [span for span in payload.get("spans", []) or [] if isinstance(span, dict)]
        index = next((i for i, span in enumerate(spans) if str(span.get("span_id")) == span_id), None)
        if index is None:
            return self._tool_error("source_span_not_found", f"Unknown span_id: {span_id}")
        radius = max(0, min(int(neighbors), 3))
        start = max(0, index - radius)
        end = min(len(spans), index + radius + 1)
        target = spans[index]
        text = str(target.get("text") or "")
        return json.dumps(
            {
                "span_id": target.get("span_id"),
                "page": target.get("page"),
                "section_name": target.get("section_name"),
                "section_type": target.get("section_type"),
                "text": text,
                "normalized_text": normalize_quote_text(text),
                "neighbors": [
                    {
                        "span_id": neighbor.get("span_id"),
                        "page": neighbor.get("page"),
                        "section_type": neighbor.get("section_type"),
                        "text": str(neighbor.get("text") or "")[:1500],
                    }
                    for neighbor in spans[start:end]
                    if neighbor is not target
                ],
            },
            indent=2,
            ensure_ascii=False,
        )

    def read_section_index(self) -> str:
        payload = json.loads(self.read_source_payload())
        return json.dumps(
            {
                "section_index": payload.get("section_index") or {},
                "extraction_quality": payload.get("extraction_quality") or {},
            },
            indent=2,
            ensure_ascii=False,
        )

    def read_skill_resource(self, path: str) -> str:
        try:
            return self.skill_pack.resource_text(path)
        except FileNotFoundError:
            return self._tool_error(
                "skill_resource_not_found",
                f"Skill resource not found: {path}. Use read_run_file for run files such as agent_schema.json.",
            )

    def read_output_schema(self) -> str:
        return self._read_file("agent_schema.json")

    def validate_agent_artifact(self, candidate_json: str) -> str:
        try:
            raw = json.loads(candidate_json)
            artifact = materialize_agent_artifact(raw)
            source_payload = None
            source_path = self.run_dir / "source_payload.json"
            if source_path.exists():
                source_payload = json.loads(source_path.read_text(encoding="utf-8"))
            issues = run_preflight(artifact, source_payload=source_payload, require_source_grounding=True)
        except Exception as exc:
            issues = [{"path": "$", "code": "invalid_json_or_schema", "message": str(exc)}]
        return json.dumps({"issue_count": len(issues), "issues": issues}, indent=2, ensure_ascii=False)

    def submit_agent_artifact(self, candidate_json: str) -> str:
        validation = json.loads(self.validate_agent_artifact(candidate_json))
        if validation["issue_count"]:
            return json.dumps({"accepted": False, "validation": validation}, indent=2, ensure_ascii=False)
        self.write_run_file("agent_output.json", json.dumps(json.loads(candidate_json), indent=2, ensure_ascii=False))
        return json.dumps({"accepted": True, "output_path": "agent_output.json"}, indent=2)

    def _read_file(self, path: str) -> str:
        try:
            return self._safe_path(path).read_text(encoding="utf-8")
        except FileNotFoundError:
            return self._tool_error("run_file_not_found", f"Run file not found: {path}")
        except ValueError as exc:
            return self._tool_error("invalid_run_file_path", str(exc))

    def _safe_path(self, path: str) -> Path:
        candidate = (self.run_dir / path).resolve()
        if candidate != self.run_dir and self.run_dir not in candidate.parents:
            raise ValueError(f"Path escapes run directory: {path}")
        return candidate

    def _tool_error(self, code: str, message: str) -> str:
        return json.dumps({"error": {"code": code, "message": message}}, indent=2, ensure_ascii=False)
