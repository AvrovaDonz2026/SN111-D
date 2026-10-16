"""Run a bounded set of saved testnet fixtures through both eligibility contracts."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if not 1 <= len(manifest) <= 5:
        parser.error('Use one to five fixtures.')
    if len({x['paper_id'] for x in manifest}) != len(manifest):
        parser.error('Use distinct paper fixtures.')
    if args.output_dir.exists():
        parser.error('Use a new suite output directory.')
    args.output_dir.mkdir(parents=True)
    status = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'papers': []}
    for index, item in enumerate(manifest):
        status['papers'].append({**item, 'contracts': ['verbose', 'compact'] if index % 2 == 0 else ['compact', 'verbose'], 'status': 'queued'})

    def save():
        pending = args.output_dir / 'suite_status.tmp'
        pending.write_text(json.dumps(status, indent=2))
        pending.replace(args.output_dir / 'suite_status.json')

    def run(index):
        item = status['papers'][index]
        command = [sys.executable, 'scripts/compare_eligibility_contracts.py', '--env-file', str(args.env_file.resolve()), '--fixture-file', str((args.manifest.parent / item['fixture_dir'] / 'fixture.json').resolve()), '--output-dir', str((args.output_dir / item['paper_id']).resolve()), '--contracts', *item['contracts']]
        with (args.output_dir / (item['paper_id'] + '.log')).open('w') as log:
            return subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode

    save()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, index): index for index in range(len(manifest))}
        for future in as_completed(futures):
            item = status['papers'][futures[future]]
            item['exit_code'] = future.result()
            item['status'] = 'complete' if item['exit_code'] == 0 else 'failed'
            item['finished_at'] = datetime.now(timezone.utc).isoformat()
            save()
            print(item['paper_id'], item['status'], flush=True)
    status['status'] = 'complete' if all(x['status'] == 'complete' for x in status['papers']) else 'failed'
    status['finished_at'] = datetime.now(timezone.utc).isoformat()
    save()
    directories = [args.output_dir / x['paper_id'] for x in status['papers'] if (args.output_dir / x['paper_id'] / 'comparison.json').exists()]
    if directories:
        subprocess.run([sys.executable, 'scripts/analyze_eligibility_comparisons.py', *map(str, directories), '--output-dir', str(args.output_dir / 'analysis')], check=True)
    return 0 if status['status'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
