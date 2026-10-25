# SN111 Two-Stage Extraction Protocol

Use this protocol for Claims subnet `agent_v1` runs. Goal order:

1. Avoid zero-score / invalid artifacts.
2. Maximize grounded **central** claim coverage (about 3–7 claims).
3. Only then optimize model strength or throughput.

## Stage A — Source map (do not emit final JSON yet)

1. Call `read_section_index` and note useful sections (ABSTRACT, METHODS, RESULTS, DISCUSSION, LIMITATIONS, TABLE, FIGURE).
2. Call `search_source_text` for contribution / result / limitation cues.
3. For each candidate support location, call `read_source_span(span_id)` and keep an evidence ledger:
   - span_id, page, section_type
   - verbatim short quote candidates
   - what proposition the span could support
4. Prefer Results / Discussion / Abstract for central claims. Do not invent claims from Methods alone.

## Stage B — Candidate claims

For each high-value claim draft:

- One atomic proposition only (do not mix result + sample size + causality).
- `statement`: mechanism or relationship under the paper's regime.
- `conditions`: regime + untested boundary.
- `falsification_criteria`: concrete counter-evidence.
- `status`: supported / partially supported / uncertain.
- Bind `proof` experiment IDs and `evidence_ids`.
- Attach `sources` / evidence `source_refs` with real `span_ids` and **verbatim** quotes.
- Mark salience in claim metadata as `central` or `supporting`. Prefer central; avoid padding with weak minors.

Target **3–7 central claims** when the paper supports them. Do not fill a 10-claim quota with weak paraphrases.

## Stage C — Deterministic audit, then JSON

Before `submit_agent_artifact`:

1. Every top-level layer exists: paper, logic, evidence, trace, src, metadata.
2. All IDs unique; every proof / evidence / verifies / linked_claim reference resolves bidirectionally.
3. Every quote is a contiguous substring of the cited span (use `read_source_span` + normalized text).
4. Numbers and strong qualifiers (significant, causal, increase, robust, first, best, p-values, n=) appear in connected sources.
5. No duplicate claim statements after whitespace/case normalization.
6. Run `validate_agent_artifact` and fix every issue with a **local repair**, not a full rewrite.

Only after Stage C passes, write `agent_output.json` / call `submit_agent_artifact`.

## Hard don'ts

- Do not paraphrase quotes.
- Do not cite span IDs you have not read.
- Do not upgrade correlation into causation without an author-stated causal claim and supporting quote.
- Do not copy other miners' logic/evidence/trace content.
- Do not mark technical extraction failure as `insufficient_information` in consensus review.
