from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from .config import AgentV1Config
from .ingest import PDF_READERS, ingest_artifact_json, ingest_pdf, ingest_text
from .runner import AgentV1Runner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the skill-capable agent_v1 miner pipeline.")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--artifact-json", type=Path)
    parser.add_argument("--text", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", choices=("dspy-react", "langchain-agent", "agent-cli"), default=None)
    parser.add_argument(
        "--pdf-reader",
        "--pdf-extraction-method",
        dest="pdf_reader",
        choices=PDF_READERS,
        default=None,
        help="PDF reader for --pdf inputs. Defaults to SUBNET_CLAIMS_PDF_READER or pdf-inspector.",
    )
    parser.add_argument("--skill-dir", type=Path)
    parser.add_argument("--max-agent-iters", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Materialize and validate local inputs/config only; never invoke a model or external agent.",
    )
    args = parser.parse_args()
    if sum(bool(value) for value in (args.pdf, args.artifact_json, args.text)) != 1:
        raise SystemExit("Provide exactly one of --pdf, --artifact-json, or --text.")

    base_dir = Path(__file__).resolve().parents[2]
    load_dotenv(Path.cwd() / ".env")
    load_dotenv(base_dir / ".env")
    load_dotenv(base_dir.parent / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    config = AgentV1Config.from_env(base_dir)
    if args.runtime:
        config.runtime = args.runtime
    if args.pdf_reader:
        config.pdf_reader = args.pdf_reader
    if args.skill_dir:
        config.skill_dir = args.skill_dir
    if args.max_agent_iters:
        config.max_agent_iters = args.max_agent_iters
    runner = AgentV1Runner(config)
    if args.dry_run:
        if args.pdf:
            if str(config.pdf_reader).strip().lower() == "grobid":
                raise SystemExit("--dry-run is offline and cannot use --pdf-reader grobid; use pdf-inspector or pypdf.")
            document = ingest_pdf(
                args.pdf,
                max_chars=None,
                reader=config.pdf_reader,
                grobid_url=config.grobid_url,
                grobid_cache_dir=config.cache_dir / "grobid",
                grobid_timeout_s=config.grobid_timeout_s,
                grobid_retries=config.grobid_retries,
                grobid_retry_wait_s=config.grobid_retry_wait_s,
            )
        elif args.artifact_json:
            document = ingest_artifact_json(args.artifact_json, max_chars=None)
        else:
            document = ingest_text(args.text, max_chars=None)
        run_dir = runner.prepare_run_inputs(document, output_dir=args.output_dir, source_artifact_path=args.pdf or args.artifact_json or args.text)
        logging.getLogger(__name__).info("dry-run prepared %s", run_dir)
        return 0
    if args.pdf:
        runner.run_from_pdf(args.pdf, output_dir=args.output_dir)
    elif args.artifact_json:
        runner.run_from_artifact_json(args.artifact_json, output_dir=args.output_dir)
    else:
        runner.run_from_text(args.text, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
