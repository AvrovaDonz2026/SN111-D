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

An offline experiment is available in `scripts/replay_equivalence_shared_coverage.py`.
It reads frozen replay records and diagnostic-corrected scores, verifies ordinary
coverage reproduction, and assigns `1/n` only to retained comparison-group members.
Groups merged into a final Silver unit use a single pool of distinct retained group
miners. Bronze and duplicate claims from one miner do not increase `n`. Nongroup-only
contributors keep full credit. The default `group-priority` policy keeps group members
fractional even if they also have another candidate in the unit; an explicit
`--mixed-contributor-policy max-credit` variant measures that alternative.
Importance denominators, minor-tier caps and diagnostic/adjudication quality stay
fixed. This script does not modify deployed scoring or submit weights.

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
miners. Judge system instructions are unchanged.

Five uncached paired testnet replays on 2026-10-18 recorded output tokens falling
from 639,093 to 475,806 (25.5%) and measured-stage total tokens from 1,406,811 to
1,098,001 (22.0%). Canonicalization total tokens increased 38.4%; all ten draft/audit
sequences completed without repair. Nine of 28 remaining jointly decided cases
changed decisions, and source-support admission differences require review.
No-bypass controls also varied, so the aggregate reduction is not a stable causal
estimate for larger mainnet panels. Keep the bypass experimental pending review.

The paired runner disables DSPy memory/disk caches and stage resume. Successful
zero-usage responses invalidate measurement. Comparison is shared between arms
and charged logically once per arm; embedding counters, extraction and reference
generation are outside these totals. The control uses the new comparator with
bypass disabled, rather than the complete pre-feature comparator contract.
A separate uncached audit-only testnet smoke completed without backend or weight
writes. Detailed results and private source artifacts are retained in Claims Team
Outputs, dated 2026-10-18, rather than committed to this repository.
