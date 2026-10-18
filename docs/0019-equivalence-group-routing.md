# Equivalence groups before eligibility adjudication

The file-agent Silver pipeline proposes groups using embedding similarity, asks
comparison to confirm strict scientific equivalence, and saves confirmed groups
separately until the normal canonicalization stage. Remaining candidates follow
the existing one- or two-candidate eligibility panel and appellate procedure.
The legacy pipeline retains its previous behavior.
Set `CLAIMS_SILVER_EQUIVALENCE_GROUP_BYPASS=false` to retain normal eligibility
routing for paired before/after replays; comparison can still return its groups,
but they grant no bypass. The default is `true`.

## Comparison and grouping

Embedding retrieval uses the existing `CLAIMS_SILVER_PAIRING_EMBEDDING_*`
configuration. Set `CLAIMS_SILVER_PAIRING_EMBEDDING_MODE=openrouter` and configure
its provider key/model as for existing pairing. Similarity proposals use a 0.90
cosine threshold and deterministic complete-link grouping: every member must
meet the threshold against every other member. Missing or failed embeddings
produce no hints; global comparison still reviews the full candidate catalog.
Similarity never grants acceptance or an adjudication bypass.

Comparison returns its existing submission/reference reviews plus
`equivalence_groups`, disjoint arrays of short candidate references. These can
include equivalent miner-only findings. Confirmation requires the same finding,
population, conditions, outcome, direction, and material qualifiers. Refinements,
split/merge relationships and partial overlaps do not establish strict equivalence.
An incomplete or rejected comparison output grants no bypass.

For compatible older outputs without explicit groups, code can also assemble an
unambiguous reference-anchored group: every member must have a direct
`semantic_equivalent` edge to the same reference, and each submission must match
only that anchor. It does not take connected-component closure. Explicit groups
take precedence. Any material incident comparison edge prevents the affected
group from bypassing adjudication, keeping the disputed candidates on the
existing eligibility route.

## Persistence and canonicalization

`comparison/equivalence_groups.json` stores schema
`claims_comparison_equivalence_groups_v1`, paper/workspace identity, stable group
IDs, member candidate IDs, supporting comparison edge IDs, and
`pending_canonical_audit` status. Original candidate records retain statements,
miner identities, evidence records and span references. Canonicalization reloads
the saved groups rather than relying on comparison's in-memory state.

Grouped candidates are removed from adjudication case construction and are not
inserted into baseline Silver. Canonicalization receives both adjudicated
candidates and the pending groups, with explicit per-candidate admission status.
Both draft and audit must review each pending member's own evidence and
scientific eligibility. They can retain, split or exclude members. A pending
group may legitimately be excluded in full, including when members are exact
restatements with individually insufficient evidence.

Only retained members supply a final unit's evidence and contributor provenance.
Pending members with no source span resolving to paper text are mandatory exclusions.
Group IDs are retained on final units, and comparison-stage routing metadata is
preserved on Silver for inspection and replay. Canonicalization failure with
pending groups fails the paper rather than producing scores from an incomplete
partition. No additional canonicalization pass or new judge is introduced.

## Scoring and consensus

Ordinary coverage and reward calculations are unchanged. Every retained miner
member can cover the resulting Silver unit. Fractional `1/n` shared coverage is a
separate scoring experiment and is not enabled by this change.

Backend consensus materialization reads actual eligibility adjudication decisions
and their original candidate findings. Bypassed groups do not fabricate panel
votes or consensus cases; their provenance is stored in Silver metadata. Existing
backend materialization/preparation contracts require no migration for this
additive metadata.

## Verification

Offline tests cover embedding-chain resistance, strict versus material relations,
miner-only groups, saved-artifact reload and scope checks, mixed and all-grouped
papers, exclusion of unsupported members, exclusion of an entire pending group,
retained evidence/provenance, and ordinary coverage for multiple contributing
miners. Judge system instructions are unchanged. Model-backed quality and token
savings must still be measured with paired replays before deployment.
