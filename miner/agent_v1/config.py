from __future__ import annotations

import os
import shlex
from pathlib import Path

from pydantic import BaseModel, Field

from .provider import normalize_provider, provider_api_base, provider_api_key_env


class AgentV1Config(BaseModel):
    base_dir: Path
    package_dir: Path
    cache_dir: Path
    output_dir: Path
    runtime: str = "dspy-react"
    skill_dir: Path
    timeout_seconds: int = Field(default=1800, ge=1)
    max_extraction_source_chars: int = Field(default=60000, ge=1)
    pdf_reader: str = "pdf-inspector"
    grobid_url: str = "http://localhost:8070/"
    grobid_timeout_s: int = Field(default=120, ge=1)
    grobid_retries: int = Field(default=3, ge=0)
    grobid_retry_wait_s: int = Field(default=2, ge=0)
    model: str = "openrouter/google/gemma-4-27b-it"
    provider: str = "openrouter"
    api_key_env: str = "OPENROUTER_API_KEY"
    api_key: str | None = None
    api_base: str = "https://openrouter.ai/api/v1"
    temperature: float = Field(default=0.2, ge=0)
    max_tokens: int = Field(default=32768, ge=1)
    max_agent_iters: int = Field(default=4, ge=1)
    max_repair_attempts: int = Field(default=3, ge=1)
    require_source_grounding: bool = True
    experiment_version: str = "sn111d-preflight-v1"
    # Stream under the hood so silent provider hangs fail on inter-chunk timeout.
    stream_completions: bool = True
    stream_stall_seconds: float = Field(default=45.0, ge=5.0)
    # Extra attempts when the provider returns overload / mid-stream / transport errors.
    overload_retries: int = Field(default=3, ge=0)
    overload_backoff_seconds: float = Field(default=5.0, ge=0.0)
    overload_backoff_max_seconds: float = Field(default=60.0, ge=1.0)
    cli_command: list[str] = []

    @classmethod
    def from_env(cls, base_dir: Path | None = None) -> "AgentV1Config":
        resolved_base_dir = base_dir or Path(__file__).resolve().parents[2]
        package_dir = Path(__file__).resolve().parent
        provider = normalize_provider(
            os.getenv("SUBNET_CLAIMS_AGENT_PROVIDER"),
            api_base=os.getenv("SUBNET_CLAIMS_AGENT_API_BASE", ""),
        )
        api_key_env = provider_api_key_env(
            provider,
            os.getenv("SUBNET_CLAIMS_AGENT_API_KEY_ENV"),
        )
        api_base = provider_api_base(
            provider,
            os.getenv("SUBNET_CLAIMS_AGENT_API_BASE"),
        )
        return cls(
            base_dir=resolved_base_dir,
            package_dir=package_dir,
            cache_dir=package_dir / ".cache",
            output_dir=package_dir / "outputs" / "agent_v1",
            runtime=os.getenv("SUBNET_CLAIMS_AGENT_RUNTIME", "dspy-react"),
            skill_dir=Path(os.getenv("SUBNET_CLAIMS_AGENT_SKILL_DIR", str(package_dir / "skills" / "compiler"))),
            timeout_seconds=int(os.getenv("SUBNET_CLAIMS_AGENT_TIMEOUT", "1800")),
            max_extraction_source_chars=int(
                os.getenv(
                    "SUBNET_CLAIMS_AGENT_MAX_EXTRACTION_SOURCE_CHARS",
                    os.getenv("SUBNET_CLAIMS_AGENT_MAX_SOURCE_CHARS", "60000"),
                )
            ),
            pdf_reader=_pdf_reader_from_env(),
            grobid_url=os.getenv("GROBID_URL", "http://localhost:8070/"),
            grobid_timeout_s=int(os.getenv("SUBNET_CLAIMS_GROBID_TIMEOUT_S", "120")),
            grobid_retries=int(os.getenv("SUBNET_CLAIMS_GROBID_RETRIES", "3")),
            grobid_retry_wait_s=int(os.getenv("SUBNET_CLAIMS_GROBID_RETRY_WAIT_S", "2")),
            model=os.getenv(
                "SUBNET_CLAIMS_AGENT_MODEL",
                os.getenv("OPENROUTER_MODEL", "openrouter/google/gemma-4-27b-it"),
            ),
            provider=provider,
            api_key_env=api_key_env,
            api_key=os.getenv(api_key_env),
            api_base=api_base,
            temperature=float(os.getenv("SUBNET_CLAIMS_AGENT_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("SUBNET_CLAIMS_AGENT_MAX_TOKENS", "32768")),
            max_agent_iters=int(os.getenv("SUBNET_CLAIMS_AGENT_MAX_ITERS", "4")),
            max_repair_attempts=int(os.getenv("SUBNET_CLAIMS_AGENT_MAX_REPAIR_ATTEMPTS", "3")),
            require_source_grounding=_env_flag("SUBNET_CLAIMS_REQUIRE_SOURCE_GROUNDING", default=True),
            experiment_version=os.getenv("SUBNET_CLAIMS_EXPERIMENT_VERSION", "sn111d-preflight-v1").strip()
            or "sn111d-preflight-v1",
            stream_completions=_env_flag("SUBNET_CLAIMS_AGENT_STREAM", default=True),
            stream_stall_seconds=float(os.getenv("SUBNET_CLAIMS_AGENT_STREAM_STALL_SECONDS", "45")),
            overload_retries=int(os.getenv("SUBNET_CLAIMS_AGENT_OVERLOAD_RETRIES", "3")),
            overload_backoff_seconds=float(os.getenv("SUBNET_CLAIMS_AGENT_OVERLOAD_BACKOFF_SECONDS", "5")),
            overload_backoff_max_seconds=float(os.getenv("SUBNET_CLAIMS_AGENT_OVERLOAD_BACKOFF_MAX_SECONDS", "60")),
            cli_command=shlex.split(os.getenv("SUBNET_CLAIMS_AGENT_CLI_COMMAND", "")),
        }

    def require_api_key(self) -> str:
        if not self.api_key:
            raise SystemExit(f"{self.api_key_env} is required for agent_v1 model runtimes.")
        return self.api_key

    def observable_snapshot(self) -> dict[str, object]:
        """Return run metadata without exposing credentials."""
        return {
            "runtime": self.runtime,
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "skill_dir": str(self.skill_dir),
            "timeout_seconds": self.timeout_seconds,
            "max_extraction_source_chars": self.max_extraction_source_chars,
            "pdf_reader": self.pdf_reader,
            "grobid_url": self.grobid_url,
            "grobid_timeout_s": self.grobid_timeout_s,
            "grobid_retries": self.grobid_retries,
            "grobid_retry_wait_s": self.grobid_retry_wait_s,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_agent_iters": self.max_agent_iters,
            "max_repair_attempts": self.max_repair_attempts,
            "require_source_grounding": self.require_source_grounding,
            "experiment_version": self.experiment_version,
            "stream_completions": self.stream_completions,
            "stream_stall_seconds": self.stream_stall_seconds,
            "overload_retries": self.overload_retries,
            "overload_backoff_seconds": self.overload_backoff_seconds,
            "overload_backoff_max_seconds": self.overload_backoff_max_seconds,
            "cli_command_configured": bool(self.cli_command),
        }


def _pdf_reader_from_env() -> str:
    return (
        os.getenv("SUBNET_CLAIMS_PDF_READER")
        or os.getenv("SUBNET_CLAIMS_PDF_EXTRACTION_METHOD")
        or os.getenv("CLAIMS_PDF_READER")
        or "pdf-inspector"
    )


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}
