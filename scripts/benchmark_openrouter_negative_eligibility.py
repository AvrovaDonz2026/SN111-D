"""One-shot BF16 OpenRouter replay of a saved negative eligibility task.

The request uses the production task, negative-judge skill, constrained output
model, and semantic validator. It uses OpenRouter JSON-object mode with the full
schema in the prompt and explicit BF16 routing. It does not run recovery retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import ValidationError

CLAIMS_ROOT = Path(__file__).resolve().parents[1]
if str(CLAIMS_ROOT) not in sys.path:
    sys.path.insert(0, str(CLAIMS_ROOT))

from validator.agent_v1.eligibility import EligibilityAdjudicationAgentOutput
from validator.agent_v1.eligibility_dspy import _constrained_output_model
from validator.agent_v1.file_agent_workflow import (
    _skill_path,
    _validate_eligibility_payload_for_task,
)


DEFAULT_MODELS = (
    "qwen/qwen3.5-27b",
    "google/gemma-4-31b-it",
    "google/gemma-4-26b-a4b-it",
    "mistralai/mistral-small-3.2-24b-instruct",
)
API_BASE = "https://openrouter.ai/api/v1"


def request_json(url: str, key: str, body: dict[str, Any] | None = None, *, timeout: int = 600) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    request = urllib.request.Request(
        url,
        headers=headers,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None,
        method="POST" if body is not None else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read(1200).decode("utf-8", "replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc


def bf16_endpoints(model: str, key: str, max_tokens: int) -> list[dict[str, Any]]:
    data = request_json(f"{API_BASE}/models/{model}/endpoints", key).get("data") or {}
    return [
        {"tag": row.get("tag"), "provider_name": row.get("provider_name"), "quantization": row.get("quantization"),
         "max_completion_tokens": row.get("max_completion_tokens"), "supports_response_format": "response_format" in (row.get("supported_parameters") or [])}
        for row in data.get("endpoints") or []
        if row.get("quantization") == "bf16"
        and "response_format" in (row.get("supported_parameters") or [])
        and int(row.get("max_completion_tokens") or 0) >= max_tokens
    ]


def run_model(model: str, task: dict[str, Any], schema: dict[str, Any], key: str, max_tokens: int, output_dir: Path, response_format: str, disable_reasoning: bool = False, provider_only: str | None = None) -> dict[str, Any]:
    name = model.replace("/", "__")
    eligible_endpoints = bf16_endpoints(model, key, max_tokens)
    result: dict[str, Any] = {
        "model": model,
        "bf16_schema_endpoints": eligible_endpoints,
        "case_count": len(task["cases"]),
        "response_format": response_format,
        "reasoning_enabled": not disable_reasoning,
        "provider_only": provider_only,
        "first_attempt_valid": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if not eligible_endpoints:
        result["status"] = "no_eligible_bf16_endpoint"
        return result

    system = (
        "Follow the Claims negative eligibility judge policy below. Return exactly one JSON object "
        "matching the supplied schema. Evaluate every case and candidate independently; never invent "
        "source span IDs.\n\n"
        + _skill_path("claims-silver-eligibility-selection-negative").read_text(encoding="utf-8")
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "task_json:\n" + json.dumps(task, ensure_ascii=False, sort_keys=True)
             + "\n\nrequired_json_schema:\n" + json.dumps(schema, ensure_ascii=False, sort_keys=True)},
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": (
            {"type": "json_schema", "json_schema": {"name": "claims_negative_eligibility", "strict": True, "schema": schema}}
            if response_format == "json_schema" else {"type": "json_object"}
        ),
        "provider": {"quantizations": ["bf16"]},
    }
    if disable_reasoning:
        body["reasoning"] = {"enabled": False}
    if provider_only:
        body["provider"]["only"] = [provider_only]
    start = time.perf_counter()
    try:
        response = request_json(f"{API_BASE}/chat/completions", key, body)
        result["duration_seconds"] = round(time.perf_counter() - start, 2)
        result["generation_id"] = response.get("id")
        result["usage"] = response.get("usage") or {}
        choice = (response.get("choices") or [{}])[0]
        result["finish_reason"] = choice.get("finish_reason")
        message = choice.get("message") or {}
        raw = message.get("content") or ""
        result["output_chars"] = len(raw)
        result["output_sha256"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        (output_dir / f"{name}.raw.txt").write_text(raw, encoding="utf-8")
        try:
            obj = json.loads(raw)
            result["json_valid"] = isinstance(obj, dict)
            if not isinstance(obj, dict):
                raise ValueError("Response is not a JSON object")
            _constrained_output_model(EligibilityAdjudicationAgentOutput, task).model_validate(obj)
            parsed = EligibilityAdjudicationAgentOutput.model_validate(obj)
            result["pydantic_valid"] = True
            _validate_eligibility_payload_for_task(parsed, task)
            result["first_attempt_valid"] = True
            result["status"] = "valid"
            result["returned_case_count"] = len(parsed.assessments)
            result["returned_candidate_count"] = sum(len(case.candidate_assessments) for case in parsed.assessments)
            result["returned_gate_count"] = sum(len(candidate.gates) for case in parsed.assessments for candidate in case.candidate_assessments)
            result["returned_atom_count"] = sum(len(candidate.claim_atom_assessments) for case in parsed.assessments for candidate in case.candidate_assessments)
            (output_dir / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        except (json.JSONDecodeError, ValidationError, ValueError, RuntimeError) as exc:
            result["status"] = "schema_or_policy_invalid"
            result["validation_error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
    except Exception as exc:
        result["duration_seconds"] = round(time.perf_counter() - start, 2)
        result["status"] = "request_failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:700]}"

    if result.get("generation_id"):
        try:
            generation = request_json(f"{API_BASE}/generation?id={urllib.parse.quote(str(result['generation_id']))}", key)
            data = generation.get("data") or {}
            result["provider_name"] = data.get("provider_name")
            result["provider_organization_id"] = data.get("provider_organization_id")
            result["total_cost"] = data.get("total_cost")
        except Exception as exc:
            result["generation_lookup_error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--response-format", choices=("json_object", "json_schema"), default="json_object")
    parser.add_argument("--disable-reasoning", action="store_true")
    parser.add_argument("--provider-only", default=None)
    args = parser.parse_args()
    load_dotenv(CLAIMS_ROOT / ".env", override=False)
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is required")
    task = json.loads(args.task.read_text(encoding="utf-8"))
    if len(task.get("cases") or []) != 12 or task.get("judge_role") != "negative":
        raise SystemExit("Task must contain exactly 12 negative-judge cases")
    schema = _constrained_output_model(EligibilityAdjudicationAgentOutput, task).model_json_schema()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "request_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    summary = {"task_path": str(args.task), "task_sha256": hashlib.sha256(args.task.read_bytes()).hexdigest(),
               "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
               "max_tokens": args.max_tokens, "response_format": args.response_format,
               "reasoning_enabled": not args.disable_reasoning, "provider_only": args.provider_only, "results": []}
    for model in [item.strip() for item in args.models.split(",") if item.strip()]:
        print(f"Running {model}", flush=True)
        result = run_model(model, task, schema, key, args.max_tokens, args.output_dir, args.response_format, args.disable_reasoning, args.provider_only)
        summary["results"].append(result)
        (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({k: result.get(k) for k in ("model", "status", "finish_reason", "duration_seconds", "usage", "provider_name", "validation_error", "error")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
