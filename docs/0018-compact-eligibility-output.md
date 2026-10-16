# Compact eligibility output and verbose paper sampling

The eligibility policy, positive and negative judge instructions, and two-stage blind appellate procedure remain unchanged. Compact mode changes the response contract and validator processing of that response. The existing verbose models, validation rules, workflow, and policy files remain available.

## Judge output

Compact positive, negative, and appellate judges return one verdict per case:

```json
{
  "decisions": {
    "k0": {
      "decision": "a",
      "reason": "B generalizes beyond the studied population.",
      "span_ids": ["paper123-span-0003"]
    }
  }
}
```

The outer mapping correlates results with the cases in the batch. Each verdict has exactly three fields. `a` and `b` denote the first and second candidate in that case, not a fixed reference/miner identity. `neither` means no candidate satisfies the eligibility policy. For a single-candidate case, only `a` and `neither` are valid.

Judges still assess all atoms and six gates. Separate output-contract instructions replace only the verbose serialization requirements. Policy files and the DSPy signature's system instruction text are not edited. Compact responses omit atom assessments, gate assessments, and repeated candidate rationales.

The validator checks exact case coverage, permitted choices, nonblank reasons, and source-span existence. When source spans are supplied, at least one citation is required, including for rejection; without source spans, a candidate cannot pass. Technical execution or schema failures are retried and isolated as `technical_unresolved`, never fabricated as a scientific rejection. Unlike verbose mode, compact mode consumes the verdict directly rather than deriving it from emitted gate and atom flags. This is an intentional change in enforcement: citation existence still does not establish semantic support.

## One verbose paper per batch

Environment-built workflows default to:

```text
CLAIMS_ELIGIBILITY_OUTPUT_CONTRACT=compact
CLAIMS_ELIGIBILITY_SAMPLE_VERBOSE_PAPER=true
```

The batch validator selects one paper from its processable paper jobs using a versioned hash of batch identity and paper identity. Hash sampling is reproducible across thread ordering and retries with the same paper set, and varies between batches. It is not a new random selection for each adjudication batch or judge. The sampled paper uses the full verbose eligibility and blind reconstruction schemas in place of compact output. Both primary judges and any appellate work for that paper use the full path. Other pipeline stages keep their configuration.

The batch outcome records `eligibility_output_sampling`, including the selected paper and whether it was scored. Failures are reported instead of silently resampling. A changed processable paper set can change the sample. One-paper batches use verbose output for their only processable paper. A sample that produces no cases has no eligibility judgments to audit.

The sample is an audit of detailed assessments, not a paired experiment: compact and verbose judgments are not both run for the sampled paper. Direct quality comparison requires a separate replay evaluation.

Set `CLAIMS_ELIGIBILITY_OUTPUT_CONTRACT=verbose` to use the preserved full path for every paper, or set `CLAIMS_ELIGIBILITY_SAMPLE_VERBOSE_PAPER=false` for all-compact operation. Direct Python construction of `FileAgentWorkflowConfig` retains its historical verbose default; callers can explicitly select compact mode. Sampling belongs to the batch validator, not to isolated `run_paper_silver_pipeline` calls.

## Blind reconstruction and appellate judgment

Compact mode retains two separate executions:

1. Blind reconstruction sees only paper context, source spans, eligibility gates, and output-contract guidance. It does not receive candidate claims, primary verdicts, or primary reasons.
2. Its output is persisted and hashed. The appellate judge then receives the locked reconstruction, disputed candidates, and compact primary decisions and reasons. It returns the same three-field verdict contract for disputed cases only.

Compact reconstruction returns:

```json
{
  "findings": [
    {
      "finding": "Among adults aged 40–65, treatment reduced systolic blood pressure by 8 mmHg compared with placebo.",
      "span_ids": ["paper123-span-0003", "paper123-span-0007"]
    }
  ],
  "packet_limitations": []
}
```

Each finding retains complete canonical wording and material qualifications. Its `span_ids` combine assertion, original-support, support-path, and salience anchors. Separate anchor categories and a duplicate qualifications field are omitted. Every qualifying finding must still be reconstructed; no count cap or ranking is introduced. The trade-off is loss of machine-readable anchor categories. Full mode preserves them. Empty findings are permitted when no qualifying finding can be reconstructed; an operational failure is handled separately.

## Consensus compatibility

Original candidate statements, source references, and source quotes are copied from validator inputs into adjudication-case findings and decision-level `candidate_findings`. They are not regenerated by judges. This applies to both compact and verbose papers.

The backend materializers read those stored inputs independently of judge verbosity. Both direct and paginated PostgreSQL readers use a shared compatibility function; SQLite supports the same new record shape. Historical `candidate_assessments` and `claim_atoms` remain readable as a fallback. Synthetic candidate-ID recovery recognizes the new findings field.

Preparation policies, quotas, leases, and source-witness certification remain unchanged. Miner assignments still expose original candidates and the source document while hiding prior verdicts. Integration tests cover compact materialization, certified synthetic generation, and preparation.

## Rollout and verification

1. Apply the backend migration `20261016_compact_eligibility` after its predecessor, and deploy compatible backend code first. The migration updates both materialization functions and adds their shared candidate reader. Its downgrade intentionally retains additive readers so compact records remain readable.
2. Deploy validator code with the desired output-contract settings. No live environment or database changes are part of the source implementation.
3. Replay reviewed papers before production rollout. Compare decision errors, citation correctness, disagreement rates, operational failures, and output tokens. Reduced field counts alone do not establish equal judgment quality or a particular savings percentage.
4. Monitor the recorded verbose sample and failures per batch. Roll back to all-verbose mode through the output-contract setting if needed; previously written compact records remain supported.

Tests use simulated judge responses and local database execution. They establish contract handling, sampling, isolation, and compatibility, not model-backed quality equivalence or measured token savings.
