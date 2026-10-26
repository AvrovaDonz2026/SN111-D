from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .base import AgentRequest, AgentResult
from .usage import usage_from_dspy_lm
from ..config import AgentV1Config
from ..provider import dspy_model_id
from ..skillpack import SkillPack
from ..tools import AgentToolbox


class DspyReActRuntime:
    runtime_name = "dspy-react"

    def __init__(self, config: AgentV1Config) -> None:
        self.config = config

    def run_skill(self, *, skill_pack: SkillPack, run_dir: Path, request: AgentRequest) -> AgentResult:
        try:
            import dspy as dspy_module
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("dspy is required for the dspy-react agent_v1 runtime.") from exc
        if not hasattr(dspy_module, "ReAct"):
            raise RuntimeError("Installed dspy does not expose dspy.ReAct; use another agent_v1 runtime.")

        started = time.time()
        toolbox = AgentToolbox(run_dir=run_dir, skill_pack=skill_pack)
        tools = [_to_dspy_tool(dspy_module, spec) for spec in toolbox.specs()]
        lm = dspy_module.LM(
            model=dspy_model_id(
                self.config.model,
                provider=self.config.provider,
                api_base=self.config.api_base,
            ),
            api_key=self.config.require_api_key(),
            api_base=self.config.api_base,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            # Repair loops must not reuse a cached empty/invalid final_json.
            cache=False,
        )

        class CompileSignature(dspy_module.Signature):
            """Run the mounted skill with tools and produce strict Claims agent JSON."""

            skill_instructions: str = dspy_module.InputField()
            request_json: str = dspy_module.InputField()
            source_payload_json: str = dspy_module.InputField()
            output_schema_json: str = dspy_module.InputField()
            coverage_requirements_json: str = dspy_module.InputField()
            validation_feedback_json: str = dspy_module.InputField()
            final_json: str = dspy_module.OutputField(desc="Strict JSON object matching the Claims agent artifact schema.")

        program = dspy_module.ReAct(CompileSignature, tools=tools, max_iters=self.config.max_agent_iters)

        def run_program():
            return program(
                skill_instructions=_runtime_instructions(skill_pack),
                request_json=json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=False),
                source_payload_json=(run_dir / request.source_payload_path).read_text(encoding="utf-8"),
                output_schema_json=(run_dir / request.output_schema_path).read_text(encoding="utf-8"),
                coverage_requirements_json=json.dumps(_coverage_requirements(request), indent=2, ensure_ascii=False),
                validation_feedback_json=(
                    (run_dir / request.validation_feedback_path).read_text(encoding="utf-8")
                    if request.validation_feedback_path
                    else "{}"
                ),
            )

        output_path = run_dir / request.expected_output_path
        # Drop stale drafts from prior repair attempts so a later final_json or
        # submit_agent_artifact can replace a previously blocked artifact.
        if output_path.exists():
            output_path.unlink()
        if hasattr(dspy_module, "context"):
            with dspy_module.context(lm=lm):
                prediction = run_program()
        else:
            dspy_module.configure(lm=lm)
            prediction = run_program()
        raw_output = str(getattr(prediction, "final_json", "") or "").strip()
        if output_path.exists():
            # submit_agent_artifact already wrote an accepted payload during the loop.
            pass
        elif raw_output:
            output_path.write_text(raw_output, encoding="utf-8")
        else:
            output_path.write_text("{}", encoding="utf-8")
        return AgentResult(
            output_path=output_path,
            manifest={
                "runtime": self.runtime_name,
                "model": self.config.model,
                "elapsed_seconds": round(time.time() - started, 3),
                "usage": usage_from_dspy_lm(lm),
                "skill": skill_pack.manifest(),
            },
        )


def _to_dspy_tool(dspy_module: Any, spec):
    if hasattr(dspy_module, "Tool"):
        return dspy_module.Tool(spec.func, name=spec.name, desc=spec.description)
    spec.func.__name__ = spec.name
    spec.func.__doc__ = spec.description
    return spec.func


def _coverage_requirements(request: AgentRequest) -> dict[str, Any]:
    paper = request.paper if isinstance(request.paper, dict) else {}
    claims_summary = [item for item in paper.get("claims_summary", []) if isinstance(item, str) and item.strip()]
    if claims_summary:
        minimum_claim_count = min(3, len(claims_summary)) if len(claims_summary) >= 3 else len(claims_summary)
    else:
        # PDF/text ingest often lacks claims_summary; still require a grounded claim set.
        minimum_claim_count = 3
    return {
        "minimum_claim_count": minimum_claim_count,
        "target_claim_count": "3-7 central claims for normal research papers when source-supported",
        "claims_summary": claims_summary,
        "validation_rule": "If minimum_claim_count is greater than 0, logic.claims must contain at least that many distinct source-grounded claims.",
        "evidence_rule": "For multi-claim artifacts, create distinct evidence records for distinct support bases and resolve every proof/evidence reference.",
    }


def _runtime_instructions(skill_pack: SkillPack) -> str:
    """Compact SN111 instructions.

    Embedding the full ARA skill pack (~95KB) into every ReAct turn crowds out
    paper evidence and causes empty/invalid artifacts on real PDFs. Keep a short
    protocol here and load long references via ``read_skill_resource`` on demand.
    """
    protocol = ""
    try:
        protocol = skill_pack.resource_text("references/sn111-two-stage-protocol.md").strip()
    except FileNotFoundError:
        protocol = "Stage A source map → Stage B candidate claims → Stage C deterministic audit → JSON."
    skill_name = skill_pack.name
    return "\n\n".join(
        [
            f"# Skill: {skill_name}",
            "You are the Claims agent_v1 miner. Produce one strict agent_output.json artifact.",
            protocol,
            "Do not invent quotes. Prefer 3-7 central claims with verbatim source quotes.",
            "Empty logic.claims is always invalid. Never return a skeleton without claims/evidence/experiments.",
            "Tools: read_section_index, search_source_text, read_source_span, read_source_payload,",
            "read_output_schema, read_skill_resource, validate_agent_artifact, submit_agent_artifact.",
            "Load longer ARA guidance only if needed via read_skill_resource(",
            "'references/validation-checklist.md' or 'references/claims-agent-v1-json-output-contract.md').",
            "Every claim proof ID must exist in logic.experiments; every evidence_id must exist in evidence.records.",
            "Every evidence/claim source quote must be a contiguous verbatim substring of the cited source span.",
            "Read coverage_requirements_json carefully and satisfy minimum_claim_count.",
            "Split evidence records by distinct support basis.",
            "If validation_feedback_json contains issues, fix every issue with a local repair.",
            "Call submit_agent_artifact with the final JSON when ready, and also return the same strict JSON in final_json.",
        ]
    )
