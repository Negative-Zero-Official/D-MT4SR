"""
Aggregate D-MT4SR / MT4SR run logs into a mean +/- std table across seeds,
and (optionally) a paired-by-seed comparison against the MT4SR baseline.

Scans a directory of run logs (the .txt files main.py writes), extracts the
FINAL test-set line from each (the last "{'Epoch': 'best', ...}" entry), groups
runs that differ only by seed, and reports mean/std per configuration.

Usage:
    python aggregate_results.py output/
    python aggregate_results.py output/ --metric MRR
    python aggregate_results.py output/ --metric NDCG@10 --csv results.csv
    python aggregate_results.py output/ --paired            # paired vs baseline
    python aggregate_results.py output/ --paired --baseline RelationAwareSASRecModel-

WHY --paired MATTERS
--------------------
The default table treats each configuration as an independent sample and
reports mean +/- std. With only 3 seeds and per-seed spreads comparable to the
effect size, that comparison is close to powerless: it discards the fact that
config A and config B were run on the SAME seeds -- the same data order and the
same initialization draw.

A paired analysis subtracts baseline_seed_k from variant_seed_k and asks
whether the DIFFERENCES are consistently positive. The shared seed-level noise
cancels, so a small but reliable improvement becomes visible at n=3 where the
unpaired table calls it inconclusive. Report the paired numbers in the paper;
keep the unpaired table for the headline metric values.

Notes:
  - Requires that run filenames encode the seed as "-seed<N>" (see args_str in
    main.py). Logs produced before that change will all collapse into one group;
    re-run them or rename the files if you need them separated.
  - The last "'Epoch': 'best'" line in each log is the test-set result (main.py
    switches args.train_matrix to test_rating_matrix before that final eval).
    Earlier "best" lines are validation.
  - Statistics here are deliberately dependency-free (no scipy), so this runs
    anywhere main.py runs.
"""
import argparse
import ast
import csv as csvmod
import math
import os
import re
import statistics
from collections import defaultdict

BEST_RE = re.compile(r"\{'Epoch': 'best'.*?\}")
SEED_RE = re.compile(r"-seed(\d+)")

# Known model-name prefixes, longest first so the Dynamic variant is matched
# before the plain RelationAware one.
MODEL_PREFIXES = [
    'DynamicRelationAwareSASRecModel',
    'RelationAwareSASRecModel',
    'DistMeanSAModel',
    'DistSAModel',
    'SASRecModel',
]


# ---------------------------------------------------------------------------
# Small dependency-free statistics helpers
# ---------------------------------------------------------------------------

def _betacf(a, b, x, itmax=200, eps=3e-16):
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def t_test_p_two_sided(t, df):
    """Two-sided p-value for Student's t with df degrees of freedom."""
    if df <= 0:
        return float('nan')
    if not math.isfinite(t):
        return 0.0
    return _betainc(0.5 * df, 0.5, df / (df + t * t))


def paired_t(deltas):
    """Returns (t_statistic, p_value) for H0: mean(deltas) == 0."""
    n = len(deltas)
    if n < 2:
        return float('nan'), float('nan')
    mean = statistics.mean(deltas)
    sd = statistics.stdev(deltas)
    if sd == 0.0:
        # All differences identical: p is 0 if nonzero, undefined if all zero.
        return ((float('inf'), 0.0) if mean != 0 else (0.0, 1.0))
    t = mean / (sd / math.sqrt(n))
    return t, t_test_p_two_sided(t, n - 1)


def sign_test_p(wins, losses):
    """Exact two-sided sign test p-value. Ties are dropped before calling."""
    n = wins + losses
    if n == 0:
        return float('nan')
    k = max(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k, n + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

def split_config_key(config_key):
    """Splits a seed-stripped run name into (model, dataset, rest).

    Falls back to (None, None, config_key) for names that don't match any
    known model prefix, so unusual filenames still aggregate (just without
    paired grouping).
    """
    for prefix in MODEL_PREFIXES:
        if config_key.startswith(prefix + '-'):
            remainder = config_key[len(prefix) + 1:]
            parts = remainder.split('-', 1)
            dataset = parts[0]
            rest = parts[1] if len(parts) > 1 else ''
            return prefix, dataset, rest
    return None, None, config_key


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


def collect(directory, metric):
    """Walks `directory`, returns (groups, skipped).

    groups: config_key -> [(seed, value)]
    """
    groups = defaultdict(list)
    skipped = []
    for root, _dirs, files in os.walk(directory):
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
            if metric not in metrics:
                skipped.append(fn + f" (no metric {metric})")
                continue
            groups[config_key].append((seed, float(metrics[metric])))
    return groups, skipped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_unpaired(groups, metric, csv_path=None):
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
    print(f"\nMetric: {metric}   (sorted best-first)\n")
    header = f"{'config'.ljust(width)}  {'n':>2}  {'mean':>8}  {'std':>7}  runs"
    print(header)
    print('-' * len(header))
    for r in rows:
        print(f"{r['config'].ljust(width)}  {r['n']:>2}  {r['mean']:>8.4f}  "
              f"{r['std']:>7.4f}  {r['values']}")

    if any(r['n'] == 1 for r in rows):
        print("\nNote: configs with n=1 have no std -- single run, treat differences "
              "smaller than the multi-seed spread as inconclusive.")

    if csv_path:
        with open(csv_path, 'w', newline='') as f:
            w = csvmod.DictWriter(
                f, fieldnames=['config', 'n', 'seeds', 'mean', 'std', 'values'])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {csv_path}")
    return rows


def print_paired(groups, metric, baseline_hint, csv_path=None):
    """Per-dataset paired-by-seed comparison of every config vs the baseline."""
    # Reorganize: dataset -> config_key -> {seed: value}
    by_dataset = defaultdict(dict)
    for config_key, entries in groups.items():
        _model, dataset, _rest = split_config_key(config_key)
        dataset = dataset or '(unknown)'
        by_dataset[dataset][config_key] = {s: v for s, v in entries if s is not None}

    csv_rows = []
    for dataset in sorted(by_dataset):
        configs = by_dataset[dataset]
        baselines = [k for k in configs if k.startswith(baseline_hint)]
        if not baselines:
            print(f"\n[{dataset}] no config matching baseline prefix "
                  f"'{baseline_hint}' -- skipping paired analysis.")
            continue
        if len(baselines) > 1:
            print(f"\n[{dataset}] WARNING: {len(baselines)} configs match the "
                  f"baseline prefix; using '{baselines[0]}'.")
        base_key = baselines[0]
        base = configs[base_key]

        print('\n' + '=' * 78)
        print(f"PAIRED vs BASELINE  |  dataset: {dataset}  |  metric: {metric}")
        print(f"baseline: {base_key}")
        print('=' * 78)

        results = []
        for config_key, values in configs.items():
            if config_key == base_key:
                continue
            shared = sorted(set(values) & set(base))
            if not shared:
                continue
            deltas = [values[s] - base[s] for s in shared]
            wins = sum(1 for d in deltas if d > 0)
            losses = sum(1 for d in deltas if d < 0)
            mean_delta = statistics.mean(deltas)
            base_mean = statistics.mean(base[s] for s in shared)
            rel = 100.0 * mean_delta / base_mean if base_mean else float('nan')
            t_stat, p_t = paired_t(deltas)
            p_sign = sign_test_p(wins, losses)
            # Label = the part of the name after the dataset, which is what
            # actually distinguishes configs within a single dataset.
            _m, _d, label = split_config_key(config_key)
            results.append({
                'dataset': dataset,
                'config': config_key,
                'label': label,
                'n': len(shared),
                'seeds': ','.join(str(s) for s in shared),
                'mean_delta': mean_delta,
                'rel_pct': rel,
                'wins': wins,
                'losses': losses,
                'deltas': ' '.join(f'{d:+.4f}' for d in deltas),
                't': t_stat,
                'p_t': p_t,
                'p_sign': p_sign,
            })

        if not results:
            print("  (no comparable configs share seeds with the baseline)")
            continue

        results.sort(key=lambda r: r['mean_delta'], reverse=True)

        # Within one dataset every config shares the same hyperparameter
        # prefix (hidden size, lr, dropout, ...). Strip the longest common
        # prefix so the label shows only what actually differs between
        # configs -- otherwise the distinguishing flags, which live at the END
        # of the name, are exactly what gets truncated away.
        labels = [r['label'] for r in results]
        prefix = os.path.commonprefix(labels) if len(labels) > 1 else ''
        prefix = prefix[:prefix.rfind('-') + 1] if '-' in prefix else ''
        for r in results:
            short = r['label'][len(prefix):] or r['label']
            r['short'] = short if len(short) <= 58 else '...' + short[-55:]

        width = min(max(len(r['short']) for r in results), 58)
        if prefix:
            print(f"(common prefix omitted from labels: {prefix})")
        header = (f"{'config'.ljust(width)}  {'n':>2}  {'W-L':>5}  "
                  f"{'mean d':>9}  {'rel%':>7}  {'p(t)':>6}  {'p(sgn)':>6}  "
                  f"per-seed deltas")
        print(header)
        print('-' * len(header))
        for r in results:
            label = r['short'][:width].ljust(width)
            print(f"{label}  {r['n']:>2}  {r['wins']:>2}-{r['losses']:<2}  "
                  f"{r['mean_delta']:>+9.4f}  {r['rel_pct']:>+6.1f}%  "
                  f"{r['p_t']:>6.3f}  {r['p_sign']:>6.3f}  {r['deltas']}")
        csv_rows.extend(results)

    print("\nReading this table:")
    print("  mean d  = mean per-seed improvement over the baseline (paired).")
    print("  W-L     = seeds where the variant won / lost. A 3-0 sweep with a")
    print("            small mean delta is better evidence than 2-1 with a large one.")
    print("  p(t)    = paired t-test.  p(sgn) = exact sign test (no normality")
    print("            assumption, but bottoms out at 0.25 for n=3, so at three")
    print("            seeds read a clean sweep as suggestive, not significant).")
    print("  At n=3 nothing here clears p<0.05 unless the effect is large AND very")
    print("  consistent. Run 5-10 seeds on whichever configs you intend to claim.")

    if csv_path and csv_rows:
        with open(csv_path, 'w', newline='') as f:
            w = csvmod.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            for r in csv_rows:
                w.writerow(r)
        print(f"\nWrote {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('directory', help='directory containing run .txt logs')
    ap.add_argument('--metric', default='MRR',
                    help="metric to summarize (e.g. MRR, NDCG@10, HIT@10). Default: MRR")
    ap.add_argument('--csv', default=None, help='optional path to write a CSV of the table')
    ap.add_argument('--paired', action='store_true',
                    help='also print a paired-by-seed comparison against the baseline')
    ap.add_argument('--paired-only', action='store_true',
                    help='print only the paired comparison, not the mean/std table')
    ap.add_argument('--baseline', default='RelationAwareSASRecModel-',
                    help="filename prefix identifying the baseline config "
                         "(default: 'RelationAwareSASRecModel-', i.e. original MT4SR)")
    args = ap.parse_args()

    groups, skipped = collect(args.directory, args.metric)

    if not groups:
        print("No parseable logs found. Check the directory and --metric name.")
        if skipped:
            print("Skipped:", *skipped, sep='\n  ')
        return

    show_paired = args.paired or args.paired_only

    if not args.paired_only:
        print_unpaired(groups, args.metric,
                       csv_path=None if show_paired else args.csv)

    if show_paired:
        print_paired(groups, args.metric, args.baseline, csv_path=args.csv)

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s) with no final 'best' line "
              "(likely still running or crashed).")


if __name__ == '__main__':
    main()
