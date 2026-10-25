# SN111-D offline simulation

This is a small, deterministic demonstration of the SN111-D paper/task flow:

1. Build one synthetic paper with source spans and three task claims.
2. Produce five synthetic miner candidates, including a duplicate, partial
   coverage, missing grounding, and untrusted provenance.
3. Apply quality, provenance, duplicate, coverage, and grounding gates.
4. Calculate validator-like score components and compare a strict gate-first
   strategy with a quality-first strategy.

The demo is intentionally isolated from `neurons/`, wallets, Bittensor RPC,
the mainnet, and TAO accounting. It uses only Python's standard library and
does not claim real network rewards.

From the repository root:

```bash
python -m demo.sn111_sim.simulate
python -m demo.sn111_sim.simulate --json
python -m unittest discover -s demo/sn111_sim -p 'test_*.py'
```

The JSON output is suitable for inspecting the synthetic gate decisions or
feeding a local experiment. It is not a production validator score report.
