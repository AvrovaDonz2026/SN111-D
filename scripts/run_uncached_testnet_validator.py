"""Run the normal audit-only testnet validator with DSPy response caches disabled."""

from __future__ import annotations

import runpy
import sys

from scripts.benchmark_equivalence_routing import disable_response_cache


def main():
    args = sys.argv[1:]
    if "--claims.audit-only" not in args:
        raise SystemExit("This benchmark wrapper requires --claims.audit-only.")
    if (
        "--subtensor.network" not in args
        or args[args.index("--subtensor.network") + 1] != "test"
    ):
        raise SystemExit("This benchmark wrapper requires --subtensor.network test.")
    disable_response_cache()
    runpy.run_module("neurons.validator", run_name="__main__")


if __name__ == "__main__":
    main()
