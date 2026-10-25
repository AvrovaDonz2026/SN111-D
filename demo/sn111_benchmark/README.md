# SN111-D agent_v1 offline benchmark

This directory contains a deterministic replay tool for comparing a baseline
`agent_v1` artifact with a candidate and an optional repaired artifact. It is
an offline diagnostic: it does not run miners or validators, contact a source,
call a model, access Bittensor, or calculate online rewards.

Run from the repository root:

```bash
python3 -m demo.sn111_benchmark.replay \
  --baseline demo/sn111_benchmark/fixtures/baseline.json \
  --candidate demo/sn111_benchmark/fixtures/candidate.json \
  --repaired demo/sn111_benchmark/fixtures/repaired.json \
  --json-out /tmp/sn111-benchmark.json \
  --csv-out /tmp/sn111-benchmark.csv
```

The JSON contains one metric row per artifact and deltas from baseline. The
CSV is a flat artifact-level export. Metrics are:

- schema and `agent_v1` validation issue counts and codes;
- claim coverage, measured as exact baseline `claim_id` recall;
- source grounding, measured from in-artifact claim source refs or evidence
  records with source refs;
- evidence-link completeness, requiring each claim evidence ID to resolve and
  link back from the evidence record;
- duplicate statement rate after whitespace/case normalization;
- latency and cost from `metadata.runtime_metrics` when present.

Missing latency or cost is reported as `unavailable`; it is never inferred.
The tool does not check whether a quote semantically supports a claim or whether
a span exists in an external source payload. Its output must not be used as a
production validator score or as a network reward formula.

Run the local unit tests with:

```bash
python3 -m unittest discover -s demo/sn111_benchmark -p 'test_*.py'
```
