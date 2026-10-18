---
name: claims-silver-canonicalizer
description: Consolidate adjudicated scientific candidates into unique, relevant, evidence-pooled Silver claim units.
---

# Claims Silver Canonicalizer

Read all accepted candidates, adjudication consensus, paper context, and source spans before producing output.

## Objective

Create one canonical Silver unit for each unique scientific claim. Prevent split, restatement, and refinement variants from receiving duplicate credit while retaining genuinely distinct results.

## Rules

- Review every supplied candidate and place it exactly once: in one canonical unit or one relevance exclusion.
- Exclude every candidate listed in `mandatory_evidence_exclusions`; it cannot receive Silver credit.
- Keep every `mandatory_same_unit_groups` group together if retained. These groups encode exact restatements or adjudicated same-unit decisions.
- Cluster transitively across the whole accepted set. Do not make isolated pairwise decisions.
- Merge logical equivalents, paraphrases, refinements that do not add a distinct result, and split/restated versions of one underlying claim.
- Keep distinct claims separate when they assert materially different results, populations, interventions, mechanisms, outcomes, or qualifiers.
- Synthesize a precise canonical statement; do not simply select a candidate because of its ordering.
- Pool evidence from every candidate assigned to the unit. The validator performs the mechanical evidence union after your grouping decision.
- Exclude claims that may be true but are trivial, incidental, or not relevant to the paper's scientific contribution. Give a concrete reason.
- Assign `central`, `supporting`, or `minor` from the claim's role in the paper, not from miner-provided metadata.
- Use the supplied short candidate references such as `c0` exactly. Completeness is proven by the final partition, so do not emit a separate reviewed-ID list.
- Write one strict JSON object matching the supplied schema to the required output file.

Do not modify input files and do not finish before the output file validates.

## Pending comparison equivalence groups

The candidate catalog may also contain members of `pending_equivalence_groups`.
Consult `candidate_admission_status`: these members have not passed eligibility
adjudication. Independently review each member's scientific eligibility and its
own linked evidence against the paper before including it. Another member's
support does not rescue an unsupported submitted claim. Comparison confirmed
meaning only; group proposals are not mandatory same-unit constraints. Split
material differences, exclude unsupported members, and retain each supported
member's identity and evidence in the final partition.
