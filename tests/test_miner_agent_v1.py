from __future__ import annotations

import json
import os
import shlex
import sqlite3
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from miner.agent_v1.artifact import materialize_agent_artifact
from miner.agent_v1.artifact_validator import validate_agent_artifact
from miner.agent_v1.config import AgentV1Config
from miner.agent_v1.ingest import (
    SOURCE_PAYLOAD_SCHEMA_VERSION,
    apply_paper_metadata_override,
    document_source_payload,
    ingest_pdf,
    ingest_text,
)
from miner.agent_v1.preflight import normalize_quote_text, quote_in_span, run_preflight
from miner.agent_v1.provider import dspy_model_id, normalize_provider, provider_api_base, provider_api_key_env
from miner.agent_v1.runner import AgentV1Runner, _runtime_source_payload_path
from miner.agent_v1.runtime.base import AgentRequest, AgentResult
from miner.agent_v1.runtime.langchain_agent import (
    _structured_payload,
    _validation_status,
)
from miner.agent_v1.runtime.subprocess_cli import SubprocessAgentRuntime
from miner.agent_v1.runtime.usage import (
    usage_from_codex_jsonl,
    usage_from_dspy_lm,
    usage_from_hermes_machine_output,
    usage_from_hermes_sessions_jsonl,
)
from miner.agent_v1.schema import agent_json_schema
from miner.agent_v1.skillpack import load_skill_pack
from miner.agent_v1.tools import AgentToolbox
from miner.agent_v1.wrappers.prompt_agent import _valid_json_file_state
from neurons.harness_profiles import resolve_agent_harness
from neurons.miner import ClaimsMiner
from neurons.protocol import ClaimExtractionSynapse
from neurons.tasks import ClaimsTask


def test_agent_v1_skillpack_preserves_all_resources() -> None:
    skill_dir = Path("miner/agent_v1/skills/compiler")
    skill_pack = load_skill_pack(skill_dir)

    assert skill_pack.name == "compiler"
    assert "SKILL.md" in skill_pack.resources
    assert "references/ara-schema.md" in skill_pack.resources
    assert "references/exploration-tree-spec.md" in skill_pack.resources
    assert "references/claims-agent-v1-json-output-contract.md" in skill_pack.resources
    assert skill_pack.sha256
    assert "Universal ARA Compiler" in skill_pack.render_for_agent()


def test_agent_harness_profile_maps_cli_and_native_runtimes() -> None:
    hermes = resolve_agent_harness(harness="hermes-cli", model="openai/gpt-5-mini")
    chutes = resolve_agent_harness(
        harness="hermes-cli",
        model="deepseek-ai/DeepSeek-V3.2-TEE",
        provider="chutes",
    )
    dspy = resolve_agent_harness(harness="dspy-react", model="openrouter/openai/gpt-5-mini")
    codex = resolve_agent_harness(harness="codex-cli", model="gpt-5.5")

    assert hermes.runtime == "agent-cli"
    assert hermes.cli_command == f"{shlex.quote(sys.executable)} -m miner.agent_v1.wrappers.hermes_prompt"
    hermes_inner = shlex.split(hermes.inner_command)
    assert Path(hermes_inner[0]).name == "hermes"
    assert hermes_inner[1:] == [
        "chat",
        "--provider",
        "openrouter",
        "-m",
        "openai/gpt-5-mini",
        "--max-turns",
        "30",
        "-q",
    ]
    assert shlex.split(chutes.inner_command)[1:] == [
        "chat",
        "--provider",
        "chutes",
        "-m",
        "deepseek-ai/DeepSeek-V3.2-TEE",
        "--max-turns",
        "30",
        "-q",
    ]

    assert dspy.runtime == "dspy-react"
    assert dspy.model == "openrouter/openai/gpt-5-mini"
    assert dspy.cli_command == ""
    assert dspy.inner_command == ""

    assert codex.runtime == "agent-cli"
    assert codex.cli_command == f"{shlex.quote(sys.executable)} -m miner.agent_v1.wrappers.codex_prompt"
    codex_inner = shlex.split(codex.inner_command)
    assert Path(codex_inner[0]).name == "codex"
    assert codex_inner[1:] == ["exec", "--model", "gpt-5.5", "--json", "--sandbox", "workspace-write", "--skip-git-repo-check"]


def test_hermes_wrapper_exits_on_valid_output_by_default(monkeypatch) -> None:
    from miner.agent_v1.wrappers import hermes_prompt

    called = {}

    def fake_prompt_agent_main() -> int:
        called["exit_on_valid_output"] = os.getenv("CLAIMS_AGENT_EXIT_ON_VALID_OUTPUT")
        return 0

    monkeypatch.delenv("CLAIMS_AGENT_EXIT_ON_VALID_OUTPUT", raising=False)
    monkeypatch.setenv("CLAIMS_AGENT_INNER_COMMAND", "hermes chat -q")
    monkeypatch.setattr(hermes_prompt, "prompt_agent_main", fake_prompt_agent_main)

    assert hermes_prompt.main() == 0
    assert called["exit_on_valid_output"] == "true"


def test_agent_v1_config_defaults_to_pdf_inspector(monkeypatch) -> None:
    monkeypatch.delenv("SUBNET_CLAIMS_PDF_READER", raising=False)
    monkeypatch.delenv("SUBNET_CLAIMS_PDF_EXTRACTION_METHOD", raising=False)
    monkeypatch.delenv("CLAIMS_PDF_READER", raising=False)

    config = AgentV1Config.from_env(Path.cwd())

    assert config.pdf_reader == "pdf-inspector"


def test_agent_v1_config_snapshot_is_observable_and_redacted(monkeypatch) -> None:
    monkeypatch.setenv("SUBNET_CLAIMS_AGENT_API_KEY", "should-not-appear")
    monkeypatch.setenv("SUBNET_CLAIMS_AGENT_CLI_COMMAND", "echo --model test-model")
    config = AgentV1Config.from_env(Path.cwd())

    snapshot = config.observable_snapshot()

    assert snapshot["model"] == config.model
    assert snapshot["timeout_seconds"] == config.timeout_seconds
    assert snapshot["cli_command_configured"] is True
    assert "should-not-appear" not in json.dumps(snapshot)


def test_agent_v1_dry_run_materializes_inputs_without_runtime(tmp_path: Path) -> None:
    text_path = tmp_path / "paper.txt"
    text_path.write_text("A local dry-run source.", encoding="utf-8")
    config = AgentV1Config.from_env(Path.cwd())
    config.skill_dir = Path("miner/agent_v1/skills/compiler")
    run_dir = tmp_path / "dry-run"

    prepared = AgentV1Runner(config).prepare_run_inputs(
        ingest_text(text_path, max_chars=None),
        output_dir=run_dir,
        source_artifact_path=text_path,
    )

    assert prepared == run_dir.absolute()
    assert (run_dir / "runtime_config.json").exists()
    assert json.loads((run_dir / "dry_run_manifest.json").read_text()) == {
        "status": "ready",
        "runtime_invoked": False,
        "input_source": "text",
    }
    assert (run_dir / "agent_schema.json").exists()
    assert (run_dir / "data" / "paper.txt").exists()


def test_agent_v1_dry_run_rejects_grobid_reader_case_insensitively(monkeypatch, tmp_path: Path) -> None:
    from miner.agent_v1 import __main__ as agent_main

    monkeypatch.setenv("SUBNET_CLAIMS_PDF_READER", "  GROBID  ")
    monkeypatch.setattr(sys, "argv", ["agent_v1", "--pdf", str(tmp_path / "input.pdf"), "--output-dir", str(tmp_path / "out"), "--dry-run"])
    monkeypatch.setattr(agent_main, "AgentV1Runner", lambda _config: object())

    try:
        agent_main.main()
    except SystemExit as exc:
        assert "offline" in str(exc)
    else:
        raise AssertionError("dry-run accepted GROBID reader from environment")


def test_agent_v1_config_resolves_chutes_credentials(monkeypatch) -> None:
    monkeypatch.setenv("SUBNET_CLAIMS_AGENT_PROVIDER", "chutes")
    monkeypatch.setenv("SUBNET_CLAIMS_AGENT_MODEL", "deepseek-ai/DeepSeek-V3.1")
    monkeypatch.setenv("CHUTES_API_KEY", "test-chutes-key")
    monkeypatch.setenv("OPENROUTER_API_BASE", "https://wrong.example/v1")
    monkeypatch.delenv("SUBNET_CLAIMS_AGENT_API_BASE", raising=False)
    monkeypatch.delenv("SUBNET_CLAIMS_AGENT_API_KEY_ENV", raising=False)

    config = AgentV1Config.from_env(Path.cwd())

    assert config.provider == "chutes"
    assert config.api_key_env == "CHUTES_API_KEY"
    assert config.api_key == "test-chutes-key"
    assert config.api_base == "https://llm.chutes.ai/v1"
    assert dspy_model_id(config.model, provider=config.provider, api_base=config.api_base) == (
        "openai/deepseek-ai/DeepSeek-V3.1"
    )


def test_agent_v1_config_resolves_cha86_credentials(monkeypatch) -> None:
    monkeypatch.setenv("SUBNET_CLAIMS_AGENT_PROVIDER", "cha86")
    monkeypatch.setenv("SUBNET_CLAIMS_AGENT_MODEL", "gpt-6-luna")
    monkeypatch.setenv("CHA86_API_KEY", "test-cha86-key")
    monkeypatch.delenv("SUBNET_CLAIMS_AGENT_API_BASE", raising=False)
    monkeypatch.delenv("SUBNET_CLAIMS_AGENT_API_KEY_ENV", raising=False)

    config = AgentV1Config.from_env(Path.cwd())

    assert config.provider == "cha86"
    assert config.api_key_env == "CHA86_API_KEY"
    assert config.api_key == "test-cha86-key"
    assert config.api_base == "https://api.cha86.dev/v1"
    assert normalize_provider(None, api_base=config.api_base) == "cha86"
    assert provider_api_base("cha86") == "https://api.cha86.dev/v1"
    assert provider_api_key_env("cha86") == "CHA86_API_KEY"
    assert dspy_model_id(config.model, provider=config.provider, api_base=config.api_base) == "openai/gpt-6-luna"


def test_dspy_chutes_model_keeps_explicit_litellm_route() -> None:
    assert dspy_model_id(
        "openai/deepseek-ai/DeepSeek-V3.1",
        provider="chutes",
    ) == "openai/deepseek-ai/DeepSeek-V3.1"
    assert dspy_model_id(
        "openai/gpt-oss-120b",
        provider="chutes",
    ) == "openai/openai/gpt-oss-120b"
    assert dspy_model_id(
        "chutes/deepseek-ai/DeepSeek-V4-Flash-0731-TEE",
        provider="chutes",
    ) == "openai/deepseek-ai/DeepSeek-V4-Flash-0731-TEE"


def test_dspy_openrouter_model_adds_litellm_route() -> None:
    assert dspy_model_id(
        "qwen/qwen3.7-flash",
        provider="openrouter",
    ) == "openrouter/qwen/qwen3.7-flash"
    assert dspy_model_id(
        "openrouter/deepseek/deepseek-v4-flash",
        provider="openrouter",
    ) == "openrouter/deepseek/deepseek-v4-flash"


def test_agent_v1_pdf_inspector_reader_outputs_markdown_page_spans(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    page = SimpleNamespace(page=0, markdown="# Title\n\nTreatment improved outcome.", needs_ocr=False)
    result = SimpleNamespace(
        pages=[page],
        pages_with_tables=[1],
        pages_with_columns=[],
        pages_needing_ocr=[],
        is_complex=True,
    )
    monkeypatch.setitem(
        sys.modules,
        "pdf_inspector",
        SimpleNamespace(extract_pages_markdown=lambda path: result),
    )

    document = ingest_pdf(pdf_path, max_chars=10_000, reader="pdf-inspector")
    payload = document_source_payload(document, max_chars=10_000)

    assert document.raw_metadata["pdf_reader"] == "pdf-inspector"
    assert document.raw_metadata["pages_with_tables"] == [1]
    assert document.spans[0].span_id == "paper-span-0001"
    assert document.spans[0].page == 1
    assert document.spans[0].text == "# Title\n\nTreatment improved outcome."
    assert document.spans[0].metadata["reader_span_id"] == "paper-p001-markdown"
    assert payload["schema_version"] == SOURCE_PAYLOAD_SCHEMA_VERSION
    assert payload["source_metadata"]["is_complex"] is True
    assert payload["source_integrity"]["truncated"] is False
    assert payload["source_integrity"]["character_count"] == len(document.spans[0].text)


def test_agent_v1_source_payload_applies_task_paper_metadata_override(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sha256.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    result = SimpleNamespace(
        pages=[SimpleNamespace(page=0, markdown="Rietveld text.")],
        pages_with_tables=[],
        pages_with_columns=[],
        pages_needing_ocr=[],
        is_complex=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "pdf_inspector",
        SimpleNamespace(extract_pages_markdown=lambda path: result),
    )

    document = ingest_pdf(pdf_path, max_chars=10_000, reader="pdf-inspector")
    document.paper.title = "Wrong Parser Title"
    apply_paper_metadata_override(
        document,
        paper_id="rietveld_et_al_2013_science",
        title="GWAS of Educational Attainment",
        source_url="https://example.test/paper.pdf",
        source_sha256="abc",
    )
    payload = document_source_payload(document, max_chars=10_000)

    assert payload["paper"]["paper_id"] == "rietveld_et_al_2013_science"
    assert payload["paper"]["title"] == "GWAS of Educational Attainment"
    assert payload["spans"][0]["paper_id"] == "rietveld_et_al_2013_science"
    assert payload["spans"][0]["span_id"] == "rietveld_et_al_2013_science-span-0001"
    assert payload["spans"][0]["metadata"]["reader_span_id"] == "rietveld_et_al_2013_science-p001-markdown"
    assert payload["source_metadata"]["paper_metadata_override"]["source_sha256"] == "abc"


def test_miner_backend_artifact_upload_returns_manifest_only() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.payloads = []

        def post_miner_artifact(self, payload):
            self.payloads.append(payload)
            return payload

    miner = object.__new__(ClaimsMiner)
    miner.backend_client = FakeBackend()
    miner.uid = 9
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_9"))
    miner.config = SimpleNamespace(claims_pipeline="agent_v1", claims_subtensor_network_arg="test")
    task = ClaimsTask.from_dict(
        {
            "task_id": "task_001",
            "run_id": "run_001",
            "batch_id": "batch_001",
            "network": "testnet",
            "paper_id": "paper_001",
        }
    )
    paper = SimpleNamespace(
        title="Paper 001",
        paper_url="https://example.test/paper.pdf",
        source_sha256="sha",
    )
    artifact = _valid_ara_payload()
    artifact["paper"]["paper_id"] = "paper_001"
    artifact["metadata"]["submission_eligible"] = True
    source_payload = {"spans": [{"span_id": "paper_001-span-0001", "text": "source"}]}

    manifest = miner._post_batch_artifact(
        task=task,
        paper=paper,
        paper_id="paper_001",
        extraction=artifact,
        source_payload=source_payload,
    )

    assert manifest is not None
    assert manifest["transport"] == "backend_artifact_v1"
    assert manifest["artifact_id"] == "artifact_run_001_uid_9_paper_001"
    assert "agent_output" not in manifest
    assert miner.backend_client.payloads[0]["agent_output"]["paper"]["paper_id"] == "paper_001"
    assert miner.backend_client.payloads[0]["source_payload"]["spans"][0]["span_id"] == "paper_001-span-0001"


def test_miner_uses_batch_scoped_artifact_id_for_canonical_task() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.payload = None

        def post_miner_artifact(self, payload):
            self.payload = payload
            return payload

    miner = object.__new__(ClaimsMiner)
    miner.backend_client = FakeBackend()
    miner.uid = 9
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_9"))
    miner.config = SimpleNamespace(claims_pipeline="agent_v1", claims_subtensor_network_arg="test")
    task = ClaimsTask.from_dict(
        {
            "task_id": "task_001",
            "run_id": "run_002",
            "batch_id": "batch_001",
            "assignment_key": "assignment_001",
            "network": "testnet",
            "paper_id": "paper_001",
        }
    )

    manifest = miner._post_batch_artifact(
        task=task,
        paper=SimpleNamespace(title="Paper", paper_url="https://example.test/paper.pdf", source_sha256="sha"),
        paper_id="paper_001",
        extraction={"paper": {"paper_id": "paper_001"}, "logic": {"claims": []}, "metadata": {"submission_eligible": True}},
        source_payload={"spans": []},
    )

    assert manifest["artifact_id"] == "artifact_batch_001_uid_9_paper_001"
    assert miner.backend_client.payload["metadata"]["artifact_scope"] == "batch"
    assert miner.backend_client.payload["metadata"]["collector_run_id"] == "run_002"


def test_miner_does_not_upload_blocked_agent_artifact() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.payloads = []

        def post_miner_artifact(self, payload):
            self.payloads.append(payload)
            return payload

    miner = object.__new__(ClaimsMiner)
    miner.backend_client = FakeBackend()
    miner.uid = 9
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_9"))
    miner.config = SimpleNamespace(claims_pipeline="agent_v1", claims_subtensor_network_arg="test")
    task = ClaimsTask.from_dict(
        {
            "task_id": "task_001",
            "run_id": "run_001",
            "batch_id": "batch_001",
            "network": "testnet",
            "paper_id": "paper_001",
        }
    )

    try:
        miner._post_batch_artifact(
            task=task,
            paper=SimpleNamespace(title="Paper", paper_url="", source_sha256="sha"),
            paper_id="paper_001",
            extraction={"metadata": {"submission_eligible": False}},
            source_payload=None,
        )
    except RuntimeError as exc:
        assert "submission gate blocked" in str(exc)
    else:
        raise AssertionError("blocked artifact was accepted for upload")
    assert miner.backend_client.payloads == []


def test_miner_does_not_upload_agent_artifact_without_submission_eligibility() -> None:
    class FakeBackend:
        def __init__(self) -> None:
            self.payloads = []

        def post_miner_artifact(self, payload):
            self.payloads.append(payload)
            return payload

    miner = object.__new__(ClaimsMiner)
    miner.backend_client = FakeBackend()
    miner.uid = 9
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_9"))
    miner.config = SimpleNamespace(claims_pipeline="agent_v1", claims_subtensor_network_arg="test")
    task = ClaimsTask.from_dict(
        {
            "task_id": "task_001",
            "run_id": "run_001",
            "batch_id": "batch_001",
            "network": "testnet",
            "paper_id": "paper_001",
        }
    )

    try:
        miner._post_batch_artifact(
            task=task,
            paper=SimpleNamespace(title="Paper", paper_url="", source_sha256="sha"),
            paper_id="paper_001",
            extraction={"paper": {"paper_id": "paper_001"}, "logic": {"claims": []}, "metadata": {}},
            source_payload=None,
        )
    except RuntimeError as exc:
        assert "submission gate blocked" in str(exc)
    else:
        raise AssertionError("artifact without explicit eligibility was accepted for upload")
    assert miner.backend_client.payloads == []


def test_miner_ignores_malformed_agent_v1_cache(tmp_path: Path) -> None:
    class Logger:
        def warning(self, _message):
            return None

        def info(self, _message):
            return None

    miner = object.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_output_dir=tmp_path, claims_pipeline="agent_v1")
    miner.bt_logging = Logger()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "bad.json").write_text(json.dumps({"logic": {"claims": []}}), encoding="utf-8")

    assert miner._read_cached_extraction("bad") is None


def test_miner_ignores_agent_v1_cache_with_non_mapping_metadata(tmp_path: Path) -> None:
    class Logger:
        def warning(self, _message):
            return None

        def info(self, _message):
            return None

    miner = object.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_output_dir=tmp_path, claims_pipeline="agent_v1")
    miner.bt_logging = Logger()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "bad-metadata.json").write_text(
        json.dumps(
            {
                "paper": {"paper_id": "paper-001"},
                "logic": {"claims": []},
                "evidence": {"records": []},
                "metadata": ["submission_eligible", False],
            }
        ),
        encoding="utf-8",
    )

    assert miner._read_cached_extraction("bad-metadata") is None


def test_miner_ignores_agent_v1_cache_without_explicit_submission_eligibility(tmp_path: Path) -> None:
    class Logger:
        def warning(self, _message):
            return None

        def info(self, _message):
            return None

    miner = object.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_output_dir=tmp_path, claims_pipeline="agent_v1")
    miner.bt_logging = Logger()
    (tmp_path / ".cache").mkdir()
    payload = _valid_ara_payload()
    payload["metadata"].pop("submission_eligible", None)
    (tmp_path / ".cache" / "missing-eligibility.json").write_text(json.dumps(payload), encoding="utf-8")

    assert miner._read_cached_extraction("missing-eligibility") is None


def test_miner_accepts_fully_valid_agent_v1_cache(tmp_path: Path) -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_output_dir=tmp_path, claims_pipeline="agent_v1")
    miner.bt_logging = SimpleNamespace(warning=lambda *_args: None, info=lambda *_args: None)
    (tmp_path / ".cache").mkdir()
    payload = _valid_ara_payload()
    payload["metadata"]["submission_eligible"] = True
    (tmp_path / ".cache" / "valid.json").write_text(json.dumps(payload), encoding="utf-8")

    assert miner._read_cached_extraction("valid") == payload


def test_miner_rejects_agent_v1_cache_with_unknown_or_invalid_fields(tmp_path: Path) -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_output_dir=tmp_path, claims_pipeline="agent_v1")
    miner.bt_logging = SimpleNamespace(warning=lambda *_args: None, info=lambda *_args: None)
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()

    unknown = _valid_ara_payload()
    unknown["metadata"]["submission_eligible"] = True
    unknown["logic"]["claims"][0]["unrecognized"] = "would be dropped"
    (cache_dir / "unknown.json").write_text(json.dumps(unknown), encoding="utf-8")

    malformed = _valid_ara_payload()
    malformed["metadata"]["submission_eligible"] = True
    malformed.pop("trace")
    (cache_dir / "malformed.json").write_text(json.dumps(malformed), encoding="utf-8")

    assert miner._read_cached_extraction("unknown") is None
    assert miner._read_cached_extraction("malformed") is None


def test_miner_ignores_agent_v1_cache_with_invalid_utf8(tmp_path: Path) -> None:
    class Logger:
        def warning(self, _message):
            return None

        def info(self, _message):
            return None

    miner = object.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_output_dir=tmp_path, claims_pipeline="agent_v1")
    miner.bt_logging = Logger()
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "invalid-utf8.json").write_bytes(b"{\xff\xfe")

    assert miner._read_cached_extraction("invalid-utf8") is None


def test_agent_v1_toolbox_validates_and_submits_artifact(tmp_path: Path) -> None:
    skill_pack = load_skill_pack(Path("miner/agent_v1/skills/compiler"))
    (tmp_path / "source_payload.json").write_text(
        json.dumps({"spans": [{"span_id": "s1", "text": "Treatment improved outcome."}]}),
        encoding="utf-8",
    )
    (tmp_path / "agent_schema.json").write_text(json.dumps(agent_json_schema()), encoding="utf-8")
    toolbox = AgentToolbox(run_dir=tmp_path, skill_pack=skill_pack)
    artifact_json = json.dumps(_valid_ara_payload(span_id="s1"))

    validation = json.loads(toolbox.validate_agent_artifact(artifact_json))
    submit = json.loads(toolbox.submit_agent_artifact(artifact_json))
    span = json.loads(toolbox.read_source_span("s1", neighbors=0))

    assert validation["issue_count"] == 0
    assert submit["accepted"] is True
    assert "Claims Agent V1 Structured Output" in toolbox.read_output_schema()
    assert (tmp_path / "agent_output.json").exists()
    assert "s1" in toolbox.search_source_text("improved")
    assert span["span_id"] == "s1"
    assert "treatment improved outcome" in span["normalized_text"]


def test_agent_v1_runner_uses_runtime_contract(monkeypatch, tmp_path: Path) -> None:
    text_path = tmp_path / "paper.txt"
    early_source = "Treatment improved outcome in the study sample. " + ("A" * 59_952)
    late_source = "Late-paper evidence must remain available to validator span resolution."
    text_path.write_text(f"{early_source}\n\n{late_source}", encoding="utf-8")
    output_dir = tmp_path / "run"
    config = AgentV1Config.from_env(Path.cwd())
    config.output_dir = tmp_path / "outputs"
    config.runtime = "dspy-react"
    config.skill_dir = Path("miner/agent_v1/skills/compiler")
    assert len(early_source) == 60_000
    config.max_extraction_source_chars = 60_000

    class FakeRuntime:
        runtime_name = "fake"

        def run_skill(self, *, skill_pack, run_dir, request):
            assert request.source_payload_path == "extraction_source_payload.json"
            extraction_payload = json.loads((run_dir / request.source_payload_path).read_text(encoding="utf-8"))
            extraction_text = "".join(span["text"] for span in extraction_payload["spans"])
            assert extraction_text == early_source
            payload = _grounded_payload_for_run(run_dir)
            (run_dir / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
            return AgentResult(
                output_path=run_dir / request.expected_output_path,
                manifest={
                    "runtime": self.runtime_name,
                    "elapsed_seconds": 1.25,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                        "cost_usd": 0.001,
                        "source": "fake",
                    },
                    "skill": skill_pack.manifest(),
                },
                stdout="ok",
                stderr="",
            )

    monkeypatch.setattr("miner.agent_v1.runner.build_agent_runtime", lambda _config: FakeRuntime())

    artifact = AgentV1Runner(config).run_from_text(text_path, output_dir=output_dir)

    assert artifact.metadata["pipeline_name"] == "agent_v1"
    assert artifact.metadata["output_schema"] == "agent_v1"
    assert (output_dir / "request.json").exists()
    assert (output_dir / "source_payload.json").exists()
    assert (output_dir / "extraction_source_payload.json").exists()
    assert (output_dir / "agent_schema.json").exists()
    assert (output_dir / "output_contract.json").exists()
    assert (output_dir / "skill_manifest.json").exists()
    assert (output_dir / "backend_manifest.json").exists()
    assert (output_dir / "agent_output.json").exists()
    assert (output_dir / "submission_gate.json").exists()
    assert (output_dir / "paper_metrics.json").exists()
    assert (output_dir / "PAPER.md").exists()
    assert json.loads((output_dir / "agent_validation_report.json").read_text())["issue_count"] == 0
    assert artifact.metadata["validation_status"] == "valid"
    assert artifact.metadata["submission_eligible"] is True
    assert json.loads((output_dir / "submission_gate.json").read_text())["status"] == "eligible"
    assert artifact.metadata["runtime_metrics"]["elapsed_seconds"] == 1.25
    assert artifact.metadata["runtime_metrics"]["token_usage"]["total_tokens"] == 15
    assert artifact.metadata["runtime_metrics"]["cost_usd"] == 0.001
    metrics = json.loads((output_dir / "paper_metrics.json").read_text(encoding="utf-8"))
    assert metrics["artifact_valid"] is True
    assert metrics["claim_count"] == 1
    full_payload = json.loads((output_dir / "source_payload.json").read_text(encoding="utf-8"))
    archived_payload = json.loads(
        (output_dir / "data" / "agent_v1_source_payload.json").read_text(encoding="utf-8")
    )
    assert late_source in "".join(span["text"] for span in full_payload["spans"])
    assert full_payload["source_integrity"]["truncated"] is False
    assert "section_index" in full_payload
    assert "extraction_quality" in full_payload
    limited_payload = json.loads((output_dir / "extraction_source_payload.json").read_text(encoding="utf-8"))
    assert limited_payload["source_integrity"]["truncated"] is True
    assert archived_payload == full_payload


def test_agent_v1_runner_marks_exhausted_repairs_as_blocked(monkeypatch, tmp_path: Path) -> None:
    text_path = tmp_path / "paper.txt"
    text_path.write_text("A local source for an invalid candidate.", encoding="utf-8")
    output_dir = tmp_path / "run"
    config = AgentV1Config.from_env(Path.cwd())
    config.max_repair_attempts = 2
    config.skill_dir = Path("miner/agent_v1/skills/compiler")

    class AlwaysInvalidRuntime:
        runtime_name = "fake"

        def run_skill(self, *, skill_pack, run_dir, request):
            payload = _valid_ara_payload()
            payload["logic"]["claims"] = []
            (run_dir / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
            return AgentResult(output_path=run_dir / request.expected_output_path, manifest={}, stdout="", stderr="")

    monkeypatch.setattr("miner.agent_v1.runner.build_agent_runtime", lambda _config: AlwaysInvalidRuntime())

    artifact = AgentV1Runner(config).run_from_text(text_path, output_dir=output_dir)

    gate = json.loads((output_dir / "submission_gate.json").read_text(encoding="utf-8"))
    assert artifact.metadata["validation_status"] == "invalid"
    assert artifact.metadata["submission_eligible"] is False
    assert gate["status"] == "blocked"
    assert gate["issue_count"] > 0
    assert "missing" in gate["issue_codes"]


def test_file_agent_runtime_receives_complete_source_payload() -> None:
    assert _runtime_source_payload_path("dspy-react") == "extraction_source_payload.json"
    assert _runtime_source_payload_path("langchain-agent") == "extraction_source_payload.json"
    assert _runtime_source_payload_path("agent-cli") == "source_payload.json"
    assert _runtime_source_payload_path("hermes-cli") == "source_payload.json"


def test_agent_cli_runtime_recovers_valid_output_on_timeout(monkeypatch, tmp_path: Path) -> None:
    config = AgentV1Config.from_env(Path.cwd())
    config.runtime = "agent-cli"
    config.cli_command = ["fake-agent"]
    config.timeout_seconds = 1
    skill_pack = load_skill_pack(Path("miner/agent_v1/skills/compiler"))
    request = AgentRequest(
        paper={"paper_id": "paper_a"},
        source_payload_path="source_payload.json",
    )
    payload = _valid_ara_payload()

    def fake_run(command, **_kwargs):
        (tmp_path / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
        raise subprocess.TimeoutExpired(command, timeout=1, output="partial stdout", stderr="")

    monkeypatch.setattr("miner.agent_v1.runtime.subprocess_cli.subprocess.run", fake_run)

    result = SubprocessAgentRuntime(config).run_skill(skill_pack=skill_pack, run_dir=tmp_path, request=request)
    manifest = json.loads((tmp_path / "backend_manifest.json").read_text(encoding="utf-8"))

    assert result.output_path == tmp_path / "agent_output.json"
    assert manifest["returncode"] == "timeout"
    assert manifest["output_recovered"] is True
    assert manifest["recovery_reason"] == "timeout_with_valid_output"


def test_agent_cli_runtime_recovers_valid_output_on_nonzero_exit(monkeypatch, tmp_path: Path) -> None:
    config = AgentV1Config.from_env(Path.cwd())
    config.runtime = "agent-cli"
    config.cli_command = ["fake-agent"]
    skill_pack = load_skill_pack(Path("miner/agent_v1/skills/compiler"))
    request = AgentRequest(
        paper={"paper_id": "paper_a"},
        source_payload_path="source_payload.json",
    )
    payload = _valid_ara_payload()

    def fake_run(command, **_kwargs):
        (tmp_path / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.CompletedProcess(command, 1, stdout="wrote output", stderr="late failure")

    monkeypatch.setattr("miner.agent_v1.runtime.subprocess_cli.subprocess.run", fake_run)

    result = SubprocessAgentRuntime(config).run_skill(skill_pack=skill_pack, run_dir=tmp_path, request=request)
    manifest = json.loads((tmp_path / "backend_manifest.json").read_text(encoding="utf-8"))

    assert result.output_path == tmp_path / "agent_output.json"
    assert manifest["returncode"] == 1
    assert manifest["output_recovered"] is True
    assert manifest["recovery_reason"] == "nonzero_exit_with_valid_output"


def test_agent_cli_runtime_recovers_hermes_usage_from_session_store(monkeypatch, tmp_path: Path) -> None:
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    db_path = hermes_home / "state.db"
    connection = sqlite3.connect(db_path)
    connection.execute(
        """
        CREATE TABLE sessions (
            cwd TEXT, started_at REAL, input_tokens INTEGER, output_tokens INTEGER,
            cache_read_tokens INTEGER, cache_write_tokens INTEGER, reasoning_tokens INTEGER,
            estimated_cost_usd REAL, actual_cost_usd REAL
        )
        """
    )
    connection.commit()
    connection.close()

    config = AgentV1Config.from_env(Path.cwd())
    config.runtime = "agent-cli"
    config.cli_command = ["hermes", "--model", "openai/gpt-5.6-luna-pro"]
    skill_pack = load_skill_pack(Path("miner/agent_v1/skills/compiler"))
    request = AgentRequest(
        paper={"paper_id": "paper_a"},
        source_payload_path="source_payload.json",
    )
    payload = _valid_ara_payload()

    def fake_run(command, **_kwargs):
        (tmp_path / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
        connection = sqlite3.connect(db_path)
        connection.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path.resolve()), time.time(), 100, 25, 40, 5, 7, 0.071, None),
        )
        connection.commit()
        connection.close()
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr("miner.agent_v1.runtime.subprocess_cli.subprocess.run", fake_run)

    result = SubprocessAgentRuntime(config).run_skill(skill_pack=skill_pack, run_dir=tmp_path, request=request)

    assert result.manifest["usage"]["cost_usd"] == 0.071
    assert result.manifest["usage"]["source"] == "hermes_sessions_export"
    assert result.manifest["usage"]["total_tokens"] == 177


def test_langchain_structured_response_payload_accepts_artifact_dict() -> None:
    payload = _valid_ara_payload()

    assert _structured_payload({"structured_response": payload}) == payload


def test_langchain_validation_rejects_message_state_output(tmp_path: Path) -> None:
    output_path = tmp_path / "agent_output.json"
    output_path.write_text(json.dumps({"messages": [], "metadata": {}}), encoding="utf-8")

    assert _validation_status(output_path).startswith("invalid_json_or_schema:")


def test_prompt_agent_output_watch_requires_agent_artifact_shape(tmp_path: Path) -> None:
    output_path = tmp_path / "agent_output.json"
    output_path.write_text(json.dumps({"messages": [], "metadata": {}}), encoding="utf-8")
    assert _valid_json_file_state(output_path) is None

    output_path.write_text(json.dumps(_valid_ara_payload()), encoding="utf-8")
    assert _valid_json_file_state(output_path) is not None


def test_agent_v1_parses_codex_jsonl_usage() -> None:
    usage = usage_from_codex_jsonl(
        "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps(
                    {
                        "type": "turn.completed",
                        "usage": {
                            "input_tokens": 100,
                            "cached_input_tokens": 80,
                            "output_tokens": 25,
                            "reasoning_output_tokens": 5,
                        },
                    }
                ),
            ]
        )
    )

    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 130
    assert usage["source"] == "codex_exec_json"


def test_agent_v1_parses_hermes_jsonl_usage() -> None:
    usage = usage_from_hermes_sessions_jsonl(
        json.dumps(
            {
                "id": "20260715_030304_738727",
                "input_tokens": 100,
                "output_tokens": 25,
                "cache_read_tokens": 40,
                "cache_write_tokens": 10,
                "reasoning_tokens": 5,
                "estimated_cost_usd": 0.123,
            }
        )
    )

    assert usage["prompt_tokens"] == 150
    assert usage["completion_tokens"] == 25
    assert usage["total_tokens"] == 180
    assert usage["cost_usd"] == 0.123
    assert usage["cost_kind"] == "estimated"
    assert usage["cache_read_tokens"] == 40
    assert usage["reasoning_tokens"] == 5
    assert usage["source"] == "hermes_sessions_export"


def test_agent_v1_parses_hermes_oneshot_usage() -> None:
    usage = usage_from_hermes_machine_output(
        "CLAIMS_HERMES_USAGE_JSON="
        + json.dumps(
            {
                "input_tokens": 100,
                "output_tokens": 25,
                "reasoning_tokens": 5,
                "cache_read_tokens": 40,
                "total_tokens": 130,
                "estimated_cost_usd": 0.123,
            }
        )
    )

    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["reasoning_tokens"] == 5
    assert usage["cache_read_tokens"] == 40
    assert usage["total_tokens"] == 130
    assert usage["cost_usd"] == 0.123
    assert usage["cost_kind"] == "estimated"
    assert usage["source"] == "hermes_oneshot"

    exported_usage = usage_from_hermes_machine_output(
        'CLAIMS_HERMES_USAGE_JSON={"prompt_tokens":50,"completion_tokens":10,'
        '"total_tokens":60,"cost_usd":0.02,"cost_kind":"estimated"}'
    )
    assert exported_usage["cost_usd"] == 0.02
    assert exported_usage["cost_kind"] == "estimated"


def test_agent_v1_parses_dspy_history_cost_and_cache_tokens() -> None:
    lm = SimpleNamespace(
        history=[
            {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
                "cost": 0.0042,
            }
        ]
    )

    usage = usage_from_dspy_lm(lm)

    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 25
    assert usage["cache_read_tokens"] == 40
    assert usage["reasoning_tokens"] == 5
    assert usage["cost_usd"] == 0.0042
    assert usage["cost_kind"] == "estimated"


def test_miner_batch_task_returns_one_article_per_paper(monkeypatch, tmp_path: Path) -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_batch_max_workers=2,
    )
    miner.bt_logging = SimpleNamespace(info=lambda *_args, **_kwargs: None, error=lambda *_args, **_kwargs: None)
    miner._model_config_fingerprint = lambda: "model"
    miner._read_cached_extraction = lambda _cache_key: None
    miner._read_cached_source_payload = lambda _cache_key: None
    calls: list[str] = []

    def fake_run_task(paper_task, *, cache_key, validator_hotkey):
        calls.append(paper_task.paper_id)
        return (
            {"paper": {"paper_id": paper_task.paper_id}, "logic": {"claims": []}, "metadata": {"cache_key": cache_key, "submission_eligible": True}},
            {"paper": {"paper_id": paper_task.paper_id}, "spans": []},
        )

    miner._run_task = fake_run_task
    task = ClaimsTask.from_dict(
        {
            "task_id": "batch_task",
            "batch_id": "batch_1",
            "papers": [
                {"paper_id": "paper_a", "title": "Paper A", "paper_url": "https://example.test/a.pdf"},
                {"paper_id": "paper_b", "title": "Paper B", "paper_url": "https://example.test/b.pdf"},
            ],
        }
    )

    articles = miner._run_batch_task(task, validator_hotkey="validator")

    assert [article["paper_id"] for article in articles] == ["paper_a", "paper_b"]
    assert [article["status"] for article in articles] == ["completed", "completed"]
    assert {article["agent_output"]["paper"]["paper_id"] for article in articles} == {"paper_a", "paper_b"}
    assert all("extraction" not in article for article in articles)
    assert all("source_payload" not in article for article in articles)
    assert sorted(calls) == ["paper_a", "paper_b"]


def test_miner_coalesces_duplicate_in_flight_paper_work() -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(claims_pipeline="agent_v1", claims_timeout=2)
    miner.bt_logging = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    cached: dict[str, dict] = {}
    calls = 0

    miner._read_cached_extraction = lambda cache_key: cached.get(cache_key)
    miner._read_cached_source_payload = lambda _cache_key: {"spans": []}

    def fake_run_task(task, *, cache_key, validator_hotkey):
        nonlocal calls
        calls += 1
        time.sleep(0.05)
        extraction = {"paper": {"paper_id": task.paper_id}, "logic": {"claims": []}}
        cached[cache_key] = extraction
        return extraction, {"spans": []}

    miner._run_task = fake_run_task
    task = ClaimsTask.from_dict({"task_id": "task_1", "paper_id": "paper_1", "paper_url": "https://example.test/1.pdf"})

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(
                lambda _index: miner._run_task_coalesced(
                    task,
                    cache_key="same-cache-key",
                    validator_hotkey="validator",
                ),
                range(2),
            )
        )

    assert calls == 1
    assert [result[0]["paper"]["paper_id"] for result in results] == ["paper_1", "paper_1"]


def test_miner_ignores_unusable_agent_v1_source_payload_cache(tmp_path: Path) -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_timeout=2,
        claims_output_dir=tmp_path,
    )
    miner.bt_logging = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
    )
    cache_key = "invalid-source-cache"
    extraction = {
        "paper": {"paper_id": "paper_1"},
        "logic": {"claims": []},
        "evidence": {"records": []},
        "metadata": {"submission_eligible": True},
    }
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    miner._cache_path(cache_key).write_text(json.dumps(extraction), encoding="utf-8")
    miner._source_payload_cache_path(cache_key).write_text(json.dumps({}), encoding="utf-8")
    calls = 0

    def fake_run_task(task, *, cache_key, validator_hotkey):
        nonlocal calls
        calls += 1
        return extraction, {"spans": [{"span_id": "s1", "paper_id": "paper_1", "text": "source"}]}

    miner._run_task = fake_run_task
    task = ClaimsTask.from_dict(
        {"task_id": "task_1", "paper_id": "paper_1", "paper_url": "https://example.test/1.pdf"}
    )

    result, source_payload = miner._run_task_coalesced(
        task,
        cache_key=cache_key,
        validator_hotkey="validator",
    )

    assert calls == 1
    assert result is extraction
    assert source_payload["spans"][0]["text"] == "source"


def test_miner_batch_forward_keeps_top_level_payload_compact(monkeypatch) -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_max_requests_per_hotkey_minute=0,
        claims_consensus_mode="compatibility",
    )
    miner.bt_logging = SimpleNamespace(info=lambda *_args, **_kwargs: None, error=lambda *_args, **_kwargs: None)
    miner._run_batch_task = lambda _task, validator_hotkey: [
        {
            "paper_id": "paper_a",
            "status": "completed",
            "agent_output": {"paper": {"paper_id": "paper_a"}, "logic": {"claims": []}},
        }
    ]
    synapse = ClaimExtractionSynapse(
        task_id="batch_task",
        batch_id="batch_1",
        papers=[{"paper_id": "paper_a", "title": "Paper A", "paper_url": "https://example.test/a.pdf"}],
    )

    result = miner.forward(synapse)

    assert result.articles[0]["paper_id"] == "paper_a"
    assert result.articles[0]["agent_output"]["paper"]["paper_id"] == "paper_a"
    assert "extraction" not in result.articles[0]
    assert result.extraction is None
    assert result.source_payload is None
    assert result.paper_id == ""


def test_miner_forward_answers_consensus_task() -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_max_requests_per_hotkey_minute=0,
        claims_consensus_mode="compatibility",
    )
    miner.uid = 42
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_42"))
    miner.bt_logging = SimpleNamespace(info=lambda *_args, **_kwargs: None, error=lambda *_args, **_kwargs: None)
    synapse = ClaimExtractionSynapse(
        task_id="consensus_task",
        task_type="agent_v1_consensus_vote",
        consensus_case_id="mcc_case_1",
        consensus_payload={
            "consensus_case_id": "mcc_case_1",
            "case_id": "case_1",
            "run_id": "run_1",
            "batch_id": "batch_1",
            "paper_id": "paper_1",
            "options": ["candidate_a", "candidate_b", "insufficient_information"],
            "case": {
                "adjudication_case": {
                    "candidate_ids": ["bronze:C01", "miner:uid_7:C02"],
                }
            },
        },
    )

    result = miner.forward(synapse)

    assert result.error == ""
    assert result.extraction is None
    assert result.consensus_vote["schema"] == "claims_miner_consensus_vote_v1"
    assert result.consensus_vote["consensus_case_id"] == "mcc_case_1"
    assert result.consensus_vote["disposition"] == "candidate_a"


def test_miner_forward_answers_complete_consensus_round() -> None:
    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_max_requests_per_hotkey_minute=0,
        claims_consensus_mode="compatibility",
    )
    miner.uid = 42
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_42"))
    miner.bt_logging = SimpleNamespace(info=lambda *_args, **_kwargs: None, error=lambda *_args, **_kwargs: None)
    synapse = ClaimExtractionSynapse(
        task_id="consensus_round",
        task_type="agent_v1_consensus_vote",
        consensus_round_id="mcr_1",
        consensus_payload={
            "round_id": "mcr_1",
            "cases": [
                {
                    "item_id": "item_a",
                    "options": ["candidate_a", "candidate_b"],
                    "case": {"adjudication_case": {"candidate_ids": ["a", "b"]}},
                },
                {
                    "item_id": "item_b",
                    "options": ["candidate_a", "candidate_b"],
                    "case": {"adjudication_case": {"candidate_ids": ["a", "b"]}},
                },
            ],
        },
    )

    result = miner.forward(synapse)

    assert result.error == ""
    assert result.consensus_vote["schema"] == "claims_miner_consensus_round_response_v1"
    assert result.consensus_vote["round_id"] == "mcr_1"
    assert [row["item_id"] for row in result.consensus_vote["responses"]] == ["item_a", "item_b"]


def test_miner_uploads_model_consensus_response_and_returns_manifest(monkeypatch) -> None:
    class FakeBackend:
        network = "testnet"

        def __init__(self) -> None:
            self.payload = None

        def post_miner_consensus_submission(self, payload):
            self.payload = payload
            return payload

    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_max_requests_per_hotkey_minute=0,
        claims_consensus_mode="model",
    )
    miner.uid = 42
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_42"))
    miner.backend_client = FakeBackend()
    miner.bt_logging = SimpleNamespace(info=lambda *_args, **_kwargs: None, error=lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        "neurons.miner.review_consensus_assignment",
        lambda _payload: {
            "schema": "claims_miner_consensus_round_response_v1",
            "round_id": "mcr_1",
            "responses": [{"item_id": "item_a", "selected_option": "candidate_a"}],
        },
    )
    synapse = ClaimExtractionSynapse(
        task_id="consensus_round",
        task_type="agent_v1_consensus_vote",
        network="testnet",
        consensus_round_id="mcr_1",
        consensus_payload={
            "round_id": "mcr_1",
            "cases": [{"item_id": "item_a", "options": ["candidate_a", "candidate_b"]}],
        },
    )

    result = miner.forward(synapse)

    assert result.error == ""
    assert result.consensus_vote["schema"] == "claims_miner_consensus_submission_manifest_v1"
    assert result.consensus_vote["submission_id"] == "consensus_mcr_1_uid_42"
    assert "responses" not in result.consensus_vote
    assert miner.backend_client.payload["responses"][0]["item_id"] == "item_a"
    assert miner.backend_client.payload["hotkey"] == "hotkey_42"


def test_miner_returns_structured_consensus_source_failure(monkeypatch) -> None:
    from miner.agent_v1.consensus_review import ConsensusSourceError

    miner = ClaimsMiner.__new__(ClaimsMiner)
    miner.config = SimpleNamespace(
        claims_pipeline="agent_v1",
        claims_max_requests_per_hotkey_minute=0,
        claims_consensus_mode="model",
    )
    miner.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="hotkey_42"))
    miner.bt_logging = SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "neurons.miner.review_consensus_assignment",
        lambda _payload: (_ for _ in ()).throw(
            ConsensusSourceError(
                paper_id="paper_1",
                source_sha256="abc123",
                cause=ValueError("missing %%EOF"),
            )
        ),
    )
    synapse = ClaimExtractionSynapse(
        task_id="consensus_round",
        task_type="agent_v1_consensus_vote",
        network="testnet",
        consensus_round_id="mcr_1",
        consensus_payload={"round_id": "mcr_1", "cases": [{"item_id": "item_a"}]},
    )

    result = miner.forward(synapse)

    assert result.consensus_vote is None
    assert json.loads(result.error)["schema"] == "claims_consensus_source_failure_v1"


def _source_ref(span_id: str, quote: str = "Treatment improved outcome.") -> dict:
    return {
        "source_id": f"src-{span_id}",
        "source_type": "span",
        "span_ids": [span_id],
        "quote": quote,
        "role": "result",
    }


def _valid_ara_payload(
    *,
    span_id: str = "paper1-span-0001",
    quote: str = "Treatment improved outcome.",
    grounded: bool = True,
) -> dict:
    refs = [_source_ref(span_id, quote)] if grounded else []
    return {
        "paper": {
            "paper_id": "paper1",
            "title": "Synthetic Paper",
            "authors": [],
            "year": 2026,
            "abstract": "Treatment improved outcome.",
            "claims_summary": ["Treatment improved outcome."],
        },
        "logic": {
            "problem_observations": ["Outcome needed testing."],
            "gaps": ["Prior evidence was incomplete."],
            "key_insight": "Direct measurement can test the outcome.",
            "assumptions": ["The study sample is relevant."],
            "claims": [
                {
                    "claim_id": "C01",
                    "statement": "Treatment improved outcome.",
                    "conditions": "Study sample conditions.",
                    "status": "supported",
                    "falsification_criteria": "Failure to improve outcome would weaken the claim.",
                    "proof": ["E01"],
                    "evidence_ids": ["EV01"],
                    "dependencies": [],
                    "sources": list(refs),
                    "metadata": {"salience": "central"},
                }
            ],
            "concepts": [],
            "experiments": [
                {
                    "experiment_id": "E01",
                    "title": "Outcome measurement",
                    "verifies": ["C01"],
                    "setup": "Study sample.",
                    "procedure": "Measure outcome after treatment.",
                    "expected_outcome": "Outcome improves.",
                    "evidence_ids": ["EV01"],
                    "run": "reported study",
                    "source_refs": list(refs),
                }
            ],
            "related_work": [],
            "constraints": [],
        },
        "evidence": {
            "records": [
                {
                    "evidence_id": "EV01",
                    "title": "Outcome result",
                    "role": "support",
                    "summary": "Treatment improved outcome.",
                    "evidence_method": "reported result",
                    "source_refs": list(refs),
                    "linked_claim_ids": ["C01"],
                    "metadata": {},
                }
            ],
            "ledger_notes": [],
        },
        "trace": {
            "node_id": "Q0",
            "node_type": "question",
            "support_level": "inferred",
            "summary": "Does treatment improve outcome?",
            "source_refs": list(refs),
            "evidence": ["C01"],
            "children": [],
        },
        "src": {"environment": ["agent_v1 test"], "artifacts": []},
        "metadata": {},
    }


def _grounded_payload_for_run(run_dir: Path, *, quote: str = "Treatment improved outcome") -> dict:
    source = json.loads((run_dir / "source_payload.json").read_text(encoding="utf-8"))
    span_id = str(source["spans"][0]["span_id"])
    return _valid_ara_payload(span_id=span_id, quote=quote)


def test_materialize_agent_artifact_coerces_cli_structured_text_fields() -> None:
    payload = _valid_ara_payload()
    payload["logic"]["experiments"][0]["setup"] = {
        "population": "Study sample",
        "condition": "Treatment arm",
    }
    payload["logic"]["claims"][0]["conditions"] = ["Study sample conditions."]

    artifact = materialize_agent_artifact(payload)

    assert artifact.logic.experiments[0].setup == '{"condition": "Treatment arm", "population": "Study sample"}'
    assert artifact.logic.claims[0].conditions == '["Study sample conditions."]'


def test_agent_v1_artifact_validator_covers_critical_structural_links() -> None:
    cases = []

    duplicate = _valid_ara_payload()
    duplicate_claim = dict(duplicate["logic"]["claims"][0])
    duplicate_claim["claim_id"] = "C02"
    duplicate["logic"]["claims"].append(duplicate_claim)
    cases.append((duplicate, "duplicate_claim"))

    claim_without_evidence = _valid_ara_payload()
    claim_without_evidence["logic"]["claims"][0]["evidence_ids"] = []
    cases.append((claim_without_evidence, "missing_claim_evidence"))

    evidence_without_claim = _valid_ara_payload()
    evidence_without_claim["evidence"]["records"][0]["linked_claim_ids"] = []
    cases.append((evidence_without_claim, "missing_evidence_claim"))

    missing_evidence_reverse = _valid_ara_payload()
    second_claim = dict(missing_evidence_reverse["logic"]["claims"][0])
    second_claim.update(claim_id="C02", statement="A separate claim without its reverse evidence link.")
    missing_evidence_reverse["logic"]["claims"].append(second_claim)
    missing_evidence_reverse["evidence"]["records"][0]["linked_claim_ids"] = ["C02"]
    cases.append((missing_evidence_reverse, "claim_evidence_link_not_bidirectional"))

    missing_claim_reverse = _valid_ara_payload()
    second_evidence = dict(missing_claim_reverse["evidence"]["records"][0])
    second_evidence.update(evidence_id="EV02", linked_claim_ids=["C01"])
    missing_claim_reverse["evidence"]["records"].append(second_evidence)
    cases.append((missing_claim_reverse, "evidence_claim_link_not_bidirectional"))

    for payload, expected_code in cases:
        artifact = materialize_agent_artifact(payload)
        assert expected_code in {issue["code"] for issue in validate_agent_artifact(artifact)}


def test_agent_v1_artifact_validator_allows_shared_evidence() -> None:
    payload = _valid_ara_payload()
    second_claim = dict(payload["logic"]["claims"][0])
    second_claim.update(
        claim_id="C02",
        statement="A distinct claim supported by the same evidence.",
        evidence_ids=["EV01"],
    )
    payload["logic"]["claims"].append(second_claim)
    payload["evidence"]["records"][0]["linked_claim_ids"].append("C02")

    assert validate_agent_artifact(materialize_agent_artifact(payload)) == []


def test_agent_v1_preflight_rejects_ungrounded_quotes() -> None:
    payload = _valid_ara_payload(span_id="span-1", quote="this quote is not in the paper")
    artifact = materialize_agent_artifact(payload)
    source_payload = {"spans": [{"span_id": "span-1", "text": "Treatment improved outcome in the sample."}]}

    issues = run_preflight(artifact, source_payload=source_payload, require_source_grounding=True)
    codes = {issue["code"] for issue in issues}

    assert "quote_not_in_source_span" in codes
    assert quote_in_span("Treatment improved", "Treatment improved outcome.")
    assert "foobar" == normalize_quote_text("foo-\nbar")


def test_agent_v1_preflight_accepts_grounded_artifact() -> None:
    payload = _valid_ara_payload(span_id="span-1", quote="Treatment improved outcome")
    artifact = materialize_agent_artifact(payload)
    source_payload = {"spans": [{"span_id": "span-1", "text": "Treatment improved outcome in the sample."}]}

    assert run_preflight(artifact, source_payload=source_payload, require_source_grounding=True) == []


def test_agent_v1_section_index_infers_results_headers(tmp_path: Path) -> None:
    path = tmp_path / "paper.txt"
    path.write_text("# Results\n\nTreatment improved outcome in the trial.\n", encoding="utf-8")
    document = ingest_text(path, max_chars=None)
    payload = document_source_payload(document, max_chars=None)

    assert "RESULTS" in payload["section_index"]["section_types_present"]
    assert payload["extraction_quality"]["empty"] is False


def test_agent_v1_runner_blocks_duplicate_claim_submission(monkeypatch, tmp_path: Path) -> None:
    text_path = tmp_path / "paper.txt"
    text_path.write_text("A local source for duplicate claims.", encoding="utf-8")
    output_dir = tmp_path / "run"
    config = AgentV1Config.from_env(Path.cwd())
    config.max_repair_attempts = 1
    config.skill_dir = Path("miner/agent_v1/skills/compiler")

    class DuplicateClaimRuntime:
        runtime_name = "fake"

        def run_skill(self, *, skill_pack, run_dir, request):
            payload = _valid_ara_payload()
            duplicate_claim = dict(payload["logic"]["claims"][0])
            duplicate_claim["claim_id"] = "C02"
            payload["logic"]["claims"].append(duplicate_claim)
            (run_dir / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
            return AgentResult(output_path=run_dir / request.expected_output_path, manifest={}, stdout="", stderr="")

    monkeypatch.setattr("miner.agent_v1.runner.build_agent_runtime", lambda _config: DuplicateClaimRuntime())

    artifact = AgentV1Runner(config).run_from_text(text_path, output_dir=output_dir)
    gate = json.loads((output_dir / "submission_gate.json").read_text(encoding="utf-8"))

    assert artifact.metadata["submission_eligible"] is False
    assert gate["status"] == "blocked"
    assert "duplicate_claim" in gate["issue_codes"]


def test_agent_v1_runner_blocks_unknown_top_level_and_nested_fields(monkeypatch, tmp_path: Path) -> None:
    for index, path in enumerate(("extra_top", "extra_nested")):
        text_path = tmp_path / f"paper-{index}.txt"
        text_path.write_text("A local source for schema validation.", encoding="utf-8")
        output_dir = tmp_path / f"run-{index}"
        config = AgentV1Config.from_env(Path.cwd())
        config.max_repair_attempts = 1
        config.skill_dir = Path("miner/agent_v1/skills/compiler")

        class UnknownFieldRuntime:
            runtime_name = "fake"

            def run_skill(self, *, skill_pack, run_dir, request):
                payload = _valid_ara_payload()
                if path == "extra_top":
                    payload["unexpected"] = "ignored by pydantic otherwise"
                else:
                    payload["logic"]["claims"][0]["sources"] = [{"source_id": "s1", "source_type": "paper", "surprise": True}]
                (run_dir / request.expected_output_path).write_text(json.dumps(payload), encoding="utf-8")
                return AgentResult(output_path=run_dir / request.expected_output_path, manifest={}, stdout="", stderr="")

        monkeypatch.setattr("miner.agent_v1.runner.build_agent_runtime", lambda _config: UnknownFieldRuntime())
        artifact = AgentV1Runner(config).run_from_text(text_path, output_dir=output_dir)
        gate = json.loads((output_dir / "submission_gate.json").read_text(encoding="utf-8"))
        report = json.loads((output_dir / "agent_validation_report.json").read_text(encoding="utf-8"))

        assert artifact.metadata["submission_eligible"] is False
        assert gate["status"] == "blocked"
        assert any(issue["code"] == "unknown_field" for issue in report["issues"])
