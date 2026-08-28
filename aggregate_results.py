"""
Aggregate D-MT4SR / MT4SR run logs into a mean +/- std table across seeds.

Scans a directory of run logs (the .txt files main.py writes), extracts the
FINAL test-set line from each (the last "{'Epoch': 'best', ...}" entry), groups
runs that differ only by seed, and reports mean/std per configuration.

Usage:
    python aggregate_results.py output/
    python aggregate_results.py output/ --metric MRR
    python aggregate_results.py output/ --metric NDCG@10 --csv results.csv

Notes:
  - Requires that run filenames encode the seed as "-seed<N>" (see args_str in
    main.py). Logs produced before that change will all collapse into one group;
    re-run them or rename the files if you need them separated.
  - The last "'Epoch': 'best'" line in each log is the test-set result (main.py
    switches args.train_matrix to test_rating_matrix before that final eval).
    Earlier "best" lines are validation.
"""
import argparse
import ast
import csv as csvmod
import os
import re
import statistics
from collections import defaultdict

BEST_RE = re.compile(r"\{'Epoch': 'best'.*?\}")
SEED_RE = re.compile(r"-seed(\d+)")


def parse_log(path):
    """Returns (config_key, seed, metrics_dict) or None if unparseable."""
    try:
        with open(path, 'r', errors='ignore') as f:
            text = f.read()
    except OSError:
        return None

    matches = BEST_RE.findall(text)
    if not matches:
        return None

    # Last 'best' block is the test-set result.
    try:
        metrics = ast.literal_eval(matches[-1])
    except (ValueError, SyntaxError):
        return None

    name = os.path.basename(path)
    if name.endswith('.txt'):
        name = name[:-4]

    seed_match = SEED_RE.search(name)
    seed = int(seed_match.group(1)) if seed_match else None
    # Config key = filename with the seed component stripped out, so runs
    # differing only by seed group together.
    config_key = SEED_RE.sub('', name)

    return config_key, seed, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', help='directory containing run .txt logs')
    ap.add_argument('--metric', default='MRR',
                    help="metric to summarize (e.g. MRR, NDCG@10, HIT@10). Default: MRR")
    ap.add_argument('--csv', default=None, help='optional path to write a CSV of the table')
    args = ap.parse_args()

    groups = defaultdict(list)  # config_key -> [(seed, value)]
    skipped = []

    for root, _dirs, files in os.walk(args.directory):
        for fn in files:
            if not fn.endswith('.txt'):
                continue
            # Logs rotated aside by main.py when a config+seed was re-run.
            # They're kept for reference but represent superseded runs, so
            # counting them would inflate n and mix stale numbers into the mean.
            if '.prev-' in fn:
                continue
            path = os.path.join(root, fn)
            parsed = parse_log(path)
            if parsed is None:
                skipped.append(fn)
                continue
            config_key, seed, metrics = parsed
            if args.metric not in metrics:
                skipped.append(fn + f" (no metric {args.metric})")
                continue
            groups[config_key].append((seed, float(metrics[args.metric])))

    if not groups:
        print("No parseable logs found. Check the directory and --metric name.")
        if skipped:
            print("Skipped:", *skipped, sep='\n  ')
        return

    rows = []
    for config_key, entries in groups.items():
        values = [v for _s, v in entries]
        seeds = sorted(s for s, _v in entries if s is not None)
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        rows.append({
            'config': config_key,
            'n': len(values),
            'seeds': ','.join(str(s) for s in seeds) if seeds else 'unknown',
            'mean': mean,
            'std': std,
            'values': ' '.join(f'{v:.4f}' for v in sorted(values, reverse=True)),
        })

    rows.sort(key=lambda r: r['mean'], reverse=True)

    width = max(len(r['config']) for r in rows)
    print(f"\nMetric: {args.metric}   (sorted best-first)\n")
    header = f"{'config'.ljust(width)}  {'n':>2}  {'mean':>8}  {'std':>7}  runs"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['config'].ljust(width)}  {r['n']:>2}  {r['mean']:>8.4f}  "
              f"{r['std']:>7.4f}  {r['values']}")

    if any(r['n'] == 1 for r in rows):
        print("\nNote: configs with n=1 have no std -- single run, treat differences "
              "smaller than the multi-seed spread as inconclusive.")

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) with no final 'best' line "
              "(likely still running or crashed).")

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            w = csvmod.DictWriter(f, fieldnames=['config', 'n', 'seeds', 'mean', 'std', 'values'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {args.csv}")


if __name__ == '__main__':
    main()
