# Claims extraction: `section_context_v1` vs `agent_v1` schemas

This updates the earlier Agent-versus-v0 note to compare the **persisted miner outputs**, not just their extraction prompts. Both are Claims-owned schemas. `agent_v1` uses an ARA-style artifact layout, but its JSON contract is defined in this repository.

Sources: [`section_context_v1/runner.py`](../miner/section_context_v1/runner.py), its [output models](../miner/section_context_v1/models.py) and [flat extraction models](../miner/section_context_v1/schema_models.py); [`agent_v1/artifact_models.py`](../miner/agent_v1/artifact_models.py), its [runner](../miner/agent_v1/runner.py), and [validator](../miner/agent_v1/artifact_validator.py). The tables enumerate every declared field of the persisted models and every top-level key assembled by the section-context runner.

**Notation:** `*` = required by that Pydantic model (no default); `?` = nullable; `=[]`, `={}`, or `=value` = default. These are model requirements, not necessarily prompt or validation requirements. “No direct field” means there is no dedicated schema slot, even if a value could be put in free-form metadata.

## Stage 1 — Main output and sidecars

| `section_context_v1` | `agent_v1` |
| --- | --- |
| Main file: `section_context_v1_output.json` | Main file: `agent_output.json` |
| `paper`, `sections[]`, `section_summaries[]`, `paper_summary`, `section_extraction_plan[]`, `claims[]`, `evidence_items[]`, `claim_evidence_links[]`, `raw_section_outputs[]` | `ara_version`, `paper`, `logic`, `evidence`, `trace`, `src`, `metadata` |
| Separate `artifact.json`: source `paper`, `spans[]`, input `claims[]`, `evidence_items[]`, `claim_evidence_links[]`, `metadata` | Separate `source_payload.json`: input paper and spans; `agent_schema.json`: generated JSON Schema for the main output |
| Also writes `extracted_claims.csv`, `manifest.json`, optionally `tei.xml` | Also writes an ARA-style Markdown tree, `manifest.json`, and `agent_validation_report.json` |

Here are the **complete top-level keys** of the two main JSON files:

| `section_context_v1_output.json` | `agent_output.json` |
| --- | --- |
| `paper: Paper` | `ara_version: string = "1.0"` |
| `sections: SectionRecord[]` | `paper: Paper*` |
| `section_summaries: SectionSummaryRecord[]` | `logic: Logic*` |
| `paper_summary: PaperSummaryRecord` | `evidence: EvidenceLayer*` |
| `section_extraction_plan: SectionExtractionDecision[]` | `trace: TraceNode*` |
| `claims: Claim[]` | `src: SrcLayer*` |
| `evidence_items: EvidenceItem[]` | `metadata: object = {}` |
| `claim_evidence_links: ClaimEvidenceLink[]` | — |
| `raw_section_outputs: object[]` | — |

The section-context envelope is assembled in the runner rather than declared as a single Pydantic model. It has **no top-level `spans` or `metadata` key**. Source spans are in the separate `artifact.json`. Each `raw_section_outputs[]` entry has `section`, `decision`, `section_summary`, `raw_output` (selected raw model JSON), and `gated_claim_count`. Only sections sent to extraction get an entry; skipped sections still appear in the inventory and plan.

## Stage 2 — Paper and source inventory

| Concept | `section_context_v1` | `agent_v1` |
| --- | --- | --- |
| Paper | `paper.paper_id*`, `doi?`, `pmid?`, `title?`, `authors=[]`, `journal?`, `year?`, `source_type?`, `open_access?` | `paper.paper_id*`, `title=""`, `authors=[]`, `year?`, `venue?`, `doi?`, `domain?`, `keywords=[]`, `abstract=""`, `claims_summary=[]` |
| Section record | `section_id*`, `paper_id*`, `section_name*`, `section_type*`, `section_source="tei_header"`, `section_title_quality="high"`, `original_section_name?`, `page_numbers=[]`, `span_ids=[]`, `text*`, `token_count=0`, `char_count=0` | No section collection in `agent_output.json` |
| Source span | `artifact.json → spans[]`: `span_id*`, `paper_id*`, `section_type?`, `section_name?`, `page?`, `char_start?`, `char_end?`, `text*`, `table_id?`, `figure_id?`, `caption_id?`, `span_type="text"` | `source_payload.json → spans[]` is a run sidecar; output objects use `SourceRef` pointers |

The flat `ExtractionArtifact` model serialized as section-context `artifact.json` has `paper*`, `spans=[]`, `claims=[]`, `evidence_items=[]`, `claim_evidence_links=[]`, `metadata={}`. Its `Span.span_type` accepts `text`, `table_cell`, `table_row`, `figure_caption`, or `other`. `artifact.json` is the supplied/ingested source artifact, while new extractions are written to `section_context_v1_output.json`.

## Stage 3 — Context and selection

| `section_context_v1` model and all fields | Closest `agent_v1` output fields |
| --- | --- |
| `SectionSummaryRecord`: `section_id*`, `section_name*`, `section_type*`, `summary_text*`, `section_role*`, `key_entities=[]`, `key_findings=[]`, `extractability_assessment=""`, `locality_confidence?` | No section-summary object. `logic.problem_observations=[]`, `gaps=[]`, `key_insight=""` and `paper.claims_summary=[]` may hold related synthesis, with different granularity. |
| `PaperSummaryRecord`: `paper_id*`, `paper_title=""`, `paper_summary*`, `main_findings=[]`, `limitations=[]`, `evidence_map=[]` | `paper.abstract=""`, `paper.claims_summary=[]`; `logic.problem_observations=[]`, `gaps=[]`, `key_insight=""`, `assumptions=[]`, `related_work=[]`, `constraints=[]`. No single paper-summary field. |
| `SectionExtractionDecision`: `section_id*`, `should_extract*`, `reason*`, `expected_claim_types=[]`, `expected_evidence_types=[]`, `likely_claim_density=""`, `likely_evidence_density=""`, `likely_context_completeness=""` | No section selection record. `trace` is an exploration tree, not an extraction plan. |
| `raw_section_outputs[]`: `section`, `decision`, `section_summary`, `raw_output`, `gated_claim_count` | No raw per-section model-call record in the output artifact. |

The section-context pipeline builds sections, summarizes them, summarizes the paper, selects sections, extracts locally, and gates claim/evidence/link rows. Its summaries orient extraction; the prompt requires each emitted claim and evidence item to be grounded in raw section text. Agent v1 uses an agent runtime to construct its artifact, then checks the artifact's structure and references. These workflow differences explain part of the output difference.

For each selected section, the raw extractor JSON is expected to contain `claims[]`, `evidence_items[]`, and `claim_evidence_links[]`. Raw links refer to `claim_index` and `evidence_index`; materialization turns those into stable IDs and persisted `ClaimEvidenceLink` rows. `raw_section_outputs[].raw_output` retains that raw JSON, including rows later removed by gating.

## Stage 4 — Claims

| Meaning | `section_context_v1` `claims[]: Claim` | `agent_v1` `logic.claims[]: Claim` |
| --- | --- | --- |
| Identity | `claim_id*`, `paper_id*` | `claim_id*`, `source_claim_id?` |
| Statement | `claim_text*` | `statement*` |
| Structured relation | `subject*`, `predicate*`, `object*` (`SemanticField` each) | No direct subject/predicate/object fields |
| Classification | `claim_kind*`, `claim_profile?`, `epistemic_status*`, `support_origin*` | `status*`; its meaning is not guaranteed to match those flat fields |
| Applicability and payload | `context={}` (`SemanticField` map), `details={}` (arbitrary JSON) | `conditions*`, `metadata={}` |
| Testability | No dedicated field | `falsification_criteria*` |
| Source grounding | `source_span_ids=[]` | `sources=[]` (`SourceRef[]`) |
| Support | Separate `claim_evidence_links[]` | `evidence_ids=[]`; `proof=[]` (experiment IDs) |
| Claim graph | No dedicated field | `dependencies=[]` (claim IDs) |
| Confidence | `extractor_confidence?` | No dedicated field |

The nested flat models are complete here:

| Model | All fields |
| --- | --- |
| `SemanticField` | `value*`, `entity_type?`, `ontology?` |
| `OntologyAnnotation` | `annotation_type*` (`mapping`/`classification`), `raw_text?`, `normalized_text?`, `mapping_status*` (`mapped`/`ambiguous`/`unresolved`/`rejected`), `candidate_mappings=[]`, `selected_mapping?`, `mapping_method?` |
| `OntologyCandidate` | `ontology_source*`, `ontology_id*`, `ontology_label*`, `match_type?`, `confidence?` |

The flat model defaults `context` and `details` to empty objects, while section-context prompts and profile gating impose stronger extraction rules. The Agent model requires the *presence* of `conditions` and `falsification_criteria` strings; its separate deterministic validator also flags empty values and unresolved proof/evidence references. The status vocabularies should not be mapped by name alone.

## Stage 5 — Evidence, links, experiments, concepts

| Meaning | `section_context_v1` | `agent_v1` |
| --- | --- | --- |
| Evidence location | `evidence_items[]: EvidenceItem` | `evidence.records[]: EvidenceRecord` |
| Identity | `evidence_id*`, `paper_id*` | `evidence_id*`, `title*` |
| Role and summary | `role*`, `summary_text*` | `role*`, `summary*` |
| Method | `evidence_method*` (`SemanticField`) | `evidence_method=""` (string) |
| Result and format | `outcome_type?`, `presentation_type?` (`SemanticField`) | `outcome_type?`, `presentation_type?` (strings) |
| Qualifiers | `context={}` (`SemanticField` map), `details={}`, `ontology?` | `metadata={}`; no typed context/details/ontology slots |
| Provenance | `source_span_ids=[]` | `source_refs=[]` (`SourceRef[]`) |
| Claim relation | Separate `claim_evidence_links[]` | `linked_claim_ids=[]` plus claim `evidence_ids=[]` |
| Ledger | No dedicated ledger notes | `evidence.ledger_notes=[]` |

Every flat `ClaimEvidenceLink` contains `link_id*`, `claim_id*`, `evidence_id*`, `relation*`, `confidence?`. Agent v1 references claims and evidence by ID, but has **no link-level `relation` or `confidence` field**.

Agent `logic.experiments[]` contains `Experiment`: `experiment_id*`, `title*`, `verifies=[]` (claim IDs), `setup*`, `procedure*`, `expected_outcome*`, `evidence_ids=[]`, `run?`, `source_refs=[]`. Agent claim `proof[]` points to experiment IDs; claim `evidence_ids[]` points to evidence record IDs. Agent `logic.concepts[]` contains `Concept`: `concept_id*`, `label*`, `definition*`, `source_refs=[]`. Section context has neither first-class experiments nor a concept list.

## Stage 6 — Provenance, trace, and runtime

| `section_context_v1` | `agent_v1` |
| --- | --- |
| Claims and evidence use `source_span_ids[]`; `artifact.json → spans[]` supplies source text. | `SourceRef`: `source_id*`, `source_type*`, `path?`, `span_ids=[]`, `quote?`, `role="input"`. Allowed roles: `input`, `result`, `method`, `interpretation`, `metadata`. Claims use `sources[]`; evidence, concepts, experiments, and trace use `source_refs[]`. |
| `section_extraction_plan[]` records section choice; `raw_section_outputs[]` records selected extraction calls. | `trace: TraceNode*` is recursive: `node_id*`, `node_type*`, `support_level="inferred"` (`explicit`/`inferred`), `summary*`, `source_refs=[]`, `evidence=[]` (IDs), `children=[]` (`TraceNode[]`). The materializer may insert a placeholder trace if the runtime omits one. |
| Main output has no `metadata` field. `artifact.json.metadata={}` belongs to the input artifact; `manifest.json` holds run counts and paths. | `metadata={}` is part of `agent_output.json`. The runner adds `pipeline_name`, `output_schema`, `runtime`, `runtime_metrics`, `skill_name`, `skill_sha256`, `source_type`, `source_path`, and `source_metadata`. These are runtime values, not fixed Pydantic fields. |
| No source/artifact layer in the main output. | `src.environment=[]`, `src.artifacts=[]` are string lists; `ara_version="1.0"` labels the artifact layout. |

For completeness, Agent `Logic` has `problem_observations=[]`, `gaps=[]`, `key_insight=""`, `assumptions=[]`, `claims=[]`, `concepts=[]`, `experiments=[]`, `related_work=[]`, `constraints=[]`. `EvidenceLayer` has `records=[]`, `ledger_notes=[]`. Root `paper`, `logic`, `evidence`, `trace`, and `src` are required by `Artifact`, although the runner materializer can supply several missing layers before model validation.

## Practical mapping and gaps

| Section-context source | Agent destination | What needs care |
| --- | --- | --- |
| `paper.paper_id`, `title`, `authors`, `year`, `doi`, `journal` | Same names except `journal → venue` | `pmid`, `source_type`, and `open_access` need metadata if retained. |
| `claims[].claim_text` | `logic.claims[].statement` | Agent `conditions` and `falsification_criteria` need interpretation or fresh extraction. |
| `subject`, `predicate`, `object`, `claim_profile`, `context`, `details` | No dedicated Agent equivalents; possibly `conditions` or `metadata` | A simple text copy loses typed relation, profile rules, and ontology. |
| `evidence_items[].summary_text` | `evidence.records[].summary` | Agent `title` must be supplied. `SemanticField.value` can seed method/outcome/presentation strings, but qualifiers and ontology need explicit preservation. |
| `claim_evidence_links[]` | `evidence_ids[]`, `linked_claim_ids[]` | Link `relation` and `confidence` have no dedicated destination; keep both ID directions consistent. |
| `source_span_ids[]` plus source `spans[]` | `SourceRef.span_ids[]`, optional `quote` | Look up source text to populate a quote and role; a span ID alone does not provide them. |
| Summaries and section plan | No one-to-one destination | Preserve as provenance if needed; a section selection decision is not an Agent trace node. |

This table is a mapping guide, **not a lossless converter**. Section context favors section-local typed claim packets and explicit relation rows. Agent v1 adds claim conditions and falsification criteria, experiments, concepts, an exploration tree, and source-reference objects. Converting a file cannot manufacture those research judgments.
