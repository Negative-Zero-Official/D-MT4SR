"""
Run the full D-MT4SR ablation suite and aggregate the results.

Runs every (dataset, config, seed) combination, records status, and calls
aggregate_results.py at the end. Safe to interrupt and re-run: completed runs
are skipped unless --force is given, so you can stop overnight and resume.

Typical use:
    python run_experiments.py --dry-run          # see the plan, run nothing
    python run_experiments.py                    # run everything
    python run_experiments.py --datasets All_Beauty
    python run_experiments.py --configs baseline v1 dynloss
    python run_experiments.py --aggregate-only   # just re-print the tables

Each run's console output goes to <output_dir>/console/<tag>.log, and the
run status ledger lives at <output_dir>/run_status.json. The model's own
metric logs are written by main.py into <output_dir> and are what
aggregate_results.py reads.
"""
import argparse
import json
import os
import subprocess
import sys
import time

# --------------------------------------------------------------------------
# Experiment definitions
#
# 'flags' are the D-MT4SR feature switches. Everything else (lr, hidden size,
# dropout, ...) is shared across configs so comparisons stay controlled --
# see SHARED_ARGS below.
#
# Note that DynamicRelationAwareSASRecModel ALWAYS uses context-conditioned
# dynamic relation weighting; it's part of the architecture, not a flag. So
# 'v1' means "dynamic weighting, nothing else added", and each other config
# is that base plus the listed flags.
# --------------------------------------------------------------------------
CONFIGS = {
    'baseline': {
        'model_name': 'RelationAwareSASRecModel',
        'flags': [],
        'desc': 'Original MT4SR',
    },
    'sasrec': {
        'model_name': 'SASRecModel',
        'flags': [],
        'desc': 'Plain SASRec (no relation pathway)',
    },
    'v1': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': [],
        'desc': 'D-MT4SR: dynamic relation weighting only',
    },
    'dynloss': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--dynamic_loss_weights'],
        'desc': 'D-MT4SR + adaptive intra/inter loss weights',
    },
    'popneg': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--popularity_neg_sampling'],
        'desc': 'D-MT4SR + popularity-aware negative sampling',
    },
    'timedecay': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--use_time_decay'],
        'desc': 'D-MT4SR + real-timestamp relation decay',
    },
    'dynloss_popneg': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--dynamic_loss_weights', '--popularity_neg_sampling'],
        'desc': 'D-MT4SR + adaptive loss weights + popularity sampling',
    },

    # ----------------------------------------------------------------------
    # D-MT4SR v2. The v1 results showed the problem was VARIANCE, not the
    # mean: on All_Beauty the dynamic gate's seed spread was 4-8x the
    # baseline's, which makes a small mean improvement undetectable. These
    # configs attack that first, then add capacity on top.
    # ----------------------------------------------------------------------
    # The control row. MT4SR with ONLY the relation-score normalization
    # applied -- no dynamic gating at all. Without this row there is no way to
    # separate "dynamic relation weighting helps" from "normalizing the
    # relation scores helps and any weighting scheme would now work". Run this
    # on all seeds; it is the comparison a reviewer will ask for first.
    'baseline_norm': {
        'model_name': 'RelationAwareSASRecModel',
        'flags': ['--rel_score_norm=std'],
        'desc': 'CONTROL: original MT4SR + relation-score normalization only',
    },
    'v2': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std'],
        'desc': 'D-MT4SR v2: residual gate + slow gate LR + relation-score norm',
    },
    # v2 without the normalization, to isolate how much of v2's effect comes
    # from the gate changes alone versus from unsaturating the attention.
    'v2_nonorm': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1'],
        'desc': 'v2 gate changes only, WITHOUT relation-score normalization',
    },
    'v2_ent': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--gate_entropy_weight=0.05', '--gate_entropy_epochs=30'],
        'desc': 'v2 + annealed gate-entropy regularization',
    },
    'v2_mask': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--gate_use_rel_mask'],
        'desc': 'v2 + observed-relation mask fed into the gate',
    },
    'v2_pair': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--gate_pairwise'],
        'desc': 'v2 + pairwise (query,key)-conditioned gate',
    },
    'v2_full': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--gate_pairwise', '--gate_use_rel_mask',
                  '--gate_entropy_weight=0.05', '--gate_entropy_epochs=30'],
        'desc': 'v2 + pairwise + relation mask + entropy regularization',
    },
    'v2_dynloss': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--dynamic_loss_weights'],
        'desc': 'v2 + adaptive intra/inter loss weights',
    },
    'timedecay_log': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--use_time_decay', '--time_decay_log'],
        'desc': 'v2 + log-compressed real-timestamp relation decay',
    },
    'popneg_mix': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--gate_residual', '--gate_lr_scale=0.1', '--rel_score_norm=std',
                  '--popularity_neg_sampling', '--popneg_mix=0.5'],
        'desc': 'v2 + 50/50 popularity/uniform negative sampling',
    },
    'v1_norm': {
        'model_name': 'DynamicRelationAwareSASRecModel',
        'flags': ['--rel_score_norm=std'],
        'desc': 'D-MT4SR v1 (free gate) + relation-score normalization',
    },
}

# Configs carrying the main positive claim -> run across all seeds.
MAIN_CONFIGS = ['baseline', 'sasrec', 'baseline_norm', 'v1', 'dynloss', 'v2',
                'v2_nonorm', 'v2_ent', 'v2_mask', 'v2_pair', 'v2_full',
                'v2_dynloss', 'v1_norm']
# Supporting / negative-result configs -> single seed is enough to report
# "we explored this and it did not improve over our base configuration".
EXTRA_CONFIGS = ['popneg', 'timedecay', 'dynloss_popneg', 'timedecay_log',
                 'popneg_mix']

# Preset for the fast go/no-go check described in the README: a short,
# epoch-capped run of baseline vs v1 vs v2 on the SAME seeds. The point is not
# to measure final quality (these runs are deliberately under-trained) but to
# see whether v2's seed spread collapses toward the baseline's, which is the
# thing v2 was built to fix and which is measurable at a fixed epoch budget.
QUICK_CONFIGS = ['baseline', 'baseline_norm', 'v1', 'v2']
QUICK_EPOCHS = 60

# Hyperparameters held constant across every run (matching the MT4SR paper's
# setup for these datasets). Changing anything here invalidates comparisons
# against previously-collected numbers.
SHARED_ARGS = [
    '--lr=0.001',
    '--hidden_size=128',
    '--max_seq_length=100',
    '--hidden_dropout_prob=0.3',
    '--num_hidden_layers=1',
    '--weight_decay=0.0',
    '--num_attention_heads=1',
    '--attention_probs_dropout_prob=0.1',
    '--rel_loss_weight=0.1',
    '--outseq_rel_loss_weight=0.05',
]

# Per-dataset extra arguments, applied to every config for that dataset so
# comparisons within a dataset stay controlled.
#
# --rel_loss_chunk_size computes the inter-sequence relation loss in chunks
# under gradient checkpointing: identical value and gradients, bounded peak
# memory, at the cost of some speed. It's only needed when the catalog is
# large enough that the (batch*seq*num_rel, item_size) logits tensor won't fit
# in GPU memory, so datasets that don't need it are left alone (an empty/absent
# entry means "run unchunked", which is both faster and the original behavior).
#
# Appliances is a mid-sized catalog (~30k products before 5-core filtering,
# comparable to All_Beauty's ~33k), so it should run unchunked on a 12 GB card
# just as All_Beauty does. If it OOMs, add an entry below, starting large --
# e.g. ['--rel_loss_chunk_size=8192'] -- and only reduce if necessary, since
# smaller chunks mean more checkpoint recomputation and slower training.
DATASET_ARGS = {
    # 'Appliances': ['--rel_loss_chunk_size=8192'],   # uncomment only if OOM
    # 'auto' resolves from the detected VRAM at run time: unchunked on an
    # 80 GB A100, 32768 on a 40 GB card, 16384 on 24 GB, 8192 on 12-16 GB.
    # On a 12 GB card that resolves to exactly the 8192 this line used to
    # hardcode, so local runs are unchanged; the point is that the same
    # command is now correct on the handoff machine too. Chunking is
    # mathematically neutral -- same loss, same gradients, bounded memory --
    # so this cannot move any metric.
    'Office_Products': ['--rel_loss_chunk_size=auto'],
}


def build_plan(datasets, configs, main_seeds, extra_seed):
    """Returns a list of run descriptors."""
    plan = []
    for dataset in datasets:
        for cfg_name in configs:
            if cfg_name not in CONFIGS:
                raise SystemExit(f"Unknown config '{cfg_name}'. "
                                 f"Known: {', '.join(CONFIGS)}")
            seeds = main_seeds if cfg_name in MAIN_CONFIGS else [extra_seed]
            for seed in seeds:
                plan.append({
                    'tag': f'{dataset}__{cfg_name}__seed{seed}',
                    'dataset': dataset,
                    'config': cfg_name,
                    'seed': seed,
                })
    return plan


def build_command(run, output_dir, extra_args):
    cfg = CONFIGS[run['config']]
    cmd = [
        sys.executable, 'main.py',
        f"--data_name={run['dataset']}",
        f"--model_name={cfg['model_name']}",
        f"--output_dir={output_dir}",
        f"--seed={run['seed']}",
    ]
    cmd += SHARED_ARGS
    cmd += DATASET_ARGS.get(run['dataset'], [])
    cmd += cfg['flags']
    cmd += extra_args
    return cmd


def load_status(path):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            print(f"Warning: could not read {path}, starting a fresh ledger.")
    return {}


def save_status(path, status):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(status, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def check_data_present(datasets):
    """Warn (don't fail) if a dataset's preprocessed .npy isn't findable."""
    missing = []
    for d in datasets:
        candidates = [
            os.path.join('..', 'data', f'{d}Partitioned_5core.npy'),
            os.path.join('data', f'{d}Partitioned_5core.npy'),
            f'{d}Partitioned_5core.npy',
        ]
        if not any(os.path.exists(c) for c in candidates):
            missing.append(d)
    if missing:
        print("WARNING: could not locate preprocessed data for: "
              + ', '.join(missing))
        print("  Expected <dataset>Partitioned_5core.npy in ./data/ or ../data/.")
        print("  Run preprocess_fromscratch.py for those datasets first "
              "(set DATASET at the top of that script).")
        print()


def format_duration(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f'{h}h{m:02d}m'
    if m:
        return f'{m}m{s:02d}s'
    return f'{s}s'


def run_aggregation(output_dir, metrics, paired=True):
    if not os.path.exists('aggregate_results.py'):
        print("\n(aggregate_results.py not found next to this script -- "
              "skipping aggregation.)")
        return
    for metric in metrics:
        print('\n' + '=' * 70)
        print(f'AGGREGATED RESULTS: {metric}')
        print('=' * 70)
        cmd = [sys.executable, 'aggregate_results.py', output_dir,
               f'--metric={metric}']
        if paired:
            cmd.append('--paired')
        subprocess.run(cmd, check=False)


def main():
    ap = argparse.ArgumentParser(
        description='Run the D-MT4SR ablation suite and aggregate results.')
    ap.add_argument('--datasets', nargs='+', default=['All_Beauty', 'Appliances'],
                    help='datasets to run (default: All_Beauty Appliances)')
    ap.add_argument('--configs', nargs='+',
                    default=MAIN_CONFIGS + EXTRA_CONFIGS,
                    help=f'configs to run. Known: {", ".join(CONFIGS)}')
    ap.add_argument('--seeds', nargs='+', type=int, default=[1, 7, 42],
                    help='seeds for the main configs (default: 1 7 42)')
    ap.add_argument('--extra-seed', type=int, default=42,
                    help='single seed used for supporting/negative configs')
    ap.add_argument('--output-dir', default='paper_runs/',
                    help='where main.py writes logs and checkpoints')
    ap.add_argument('--metrics', nargs='+', default=['MRR', 'NDCG@10', 'HIT@10'],
                    help='metrics to aggregate at the end')
    ap.add_argument('--dry-run', action='store_true',
                    help='print the plan and the exact commands, run nothing')
    ap.add_argument('--force', action='store_true',
                    help='re-run even runs already marked completed')
    ap.add_argument('--aggregate-only', action='store_true',
                    help='skip training, just aggregate existing logs')
    ap.add_argument('--stop-on-failure', action='store_true',
                    help='abort the suite if any run fails (default: continue)')
    ap.add_argument('--quick', action='store_true',
                    help='fast go/no-go check: baseline vs v1 vs v2 on one dataset, '
                         f'capped at {QUICK_EPOCHS} epochs, into quick_check/ by default. '
                         'Deliberately under-trained -- read the SEED SPREAD, not the '
                         'absolute numbers.')
    ap.add_argument('--paired', action='store_true', default=True,
                    help='include the paired-by-seed comparison in the aggregation '
                         '(default: on)')
    ap.add_argument('--no-paired', dest='paired', action='store_false',
                    help='suppress the paired comparison')
    ap.add_argument('--passthrough', nargs=argparse.REMAINDER, default=[],
                    help='extra args forwarded verbatim to main.py '
                         '(e.g. --passthrough --epochs=50)')
    args = ap.parse_args()

    if args.quick:
        # Only override what the user didn't explicitly set. argparse doesn't
        # tell us that directly, so compare against the declared defaults.
        if args.configs == MAIN_CONFIGS + EXTRA_CONFIGS:
            args.configs = list(QUICK_CONFIGS)
        if args.output_dir == 'paper_runs/':
            # IMPORTANT: never write quick runs into paper_runs/. main.py
            # rotates any existing log for the same config+seed aside to
            # *.prev-*.txt, and aggregate_results.py ignores those -- so a
            # quick run in paper_runs/ would silently retire the full-length
            # results already collected there.
            args.output_dir = 'quick_check/'
        if args.datasets == ['All_Beauty', 'Appliances']:
            args.datasets = ['All_Beauty']
        if not any(a.startswith('--epochs') for a in args.passthrough):
            args.passthrough = list(args.passthrough) + [f'--epochs={QUICK_EPOCHS}']
        print(f'\n*** QUICK CHECK MODE ***')
        print(f'  configs : {", ".join(args.configs)}')
        print(f'  dataset : {", ".join(args.datasets)}')
        print(f'  seeds   : {args.seeds}')
        print(f'  epochs  : capped at {QUICK_EPOCHS} (runs are UNDER-TRAINED on purpose)')
        print(f'  output  : {args.output_dir}  (kept separate from paper_runs/)')
        print('  Read the per-seed spread and the paired deltas, not the absolute')
        print('  metric values -- 60 epochs is not a converged model.\n')

    output_dir = args.output_dir
    console_dir = os.path.join(output_dir, 'console')
    status_path = os.path.join(output_dir, 'run_status.json')

    if args.aggregate_only:
        run_aggregation(output_dir, args.metrics, paired=args.paired)
        return

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(console_dir, exist_ok=True)

    plan = build_plan(args.datasets, args.configs, args.seeds, args.extra_seed)
    status = load_status(status_path)

    def needs_run(r):
        if args.force:
            return True
        entry = status.get(r['tag'], {})
        if entry.get('state') != 'completed':
            return True
        # A completed run only counts if it was produced by the SAME command.
        # Otherwise a short trial (e.g. --passthrough --epochs=3) would be
        # mistaken for a finished full run and silently skipped, putting
        # trial-length numbers into the final table.
        planned = ' '.join(build_command(r, output_dir, args.passthrough))
        return entry.get('command') != planned

    todo = [r for r in plan if needs_run(r)]
    stale = [r for r in plan
             if status.get(r['tag'], {}).get('state') == 'completed'
             and r in todo and not args.force]
    already = len(plan) - len(todo)

    print(f'\nPlanned runs: {len(plan)}  '
          f'(already completed: {already}, to run now: {len(todo)})')
    print(f'Datasets: {", ".join(args.datasets)}')
    print(f'Main configs (all seeds {args.seeds}): '
          f'{", ".join(c for c in args.configs if c in MAIN_CONFIGS)}')
    extras = [c for c in args.configs if c in EXTRA_CONFIGS]
    if extras:
        print(f'Supporting configs (seed {args.extra_seed} only): {", ".join(extras)}')
    print(f'Output dir: {output_dir}\n')

    if stale:
        print(f'NOTE: {len(stale)} run(s) were previously completed with a '
              'DIFFERENT command (e.g. a shorter trial via --passthrough).')
        print('      They will be re-run so trial-length results do not end up '
              'in the final table.')
        print('      Their old logs are rotated aside to *.prev-*.txt and are '
              'excluded from aggregation.\n')

    check_data_present(args.datasets)

    if args.dry_run:
        print('--- DRY RUN: commands that would execute ---\n')
        for r in todo:
            cmd = build_command(r, output_dir, args.passthrough)
            print(f"[{r['tag']}]")
            print('  ' + ' '.join(cmd) + '\n')
        print(f'{len(todo)} run(s) would execute. '
              'Each full run can take a while -- consider running one '
              'dataset at a time.')
        return

    if not todo:
        print('Nothing to run. Aggregating existing results.\n')
        run_aggregation(output_dir, args.metrics, paired=args.paired)
        return

    suite_start = time.time()
    failures = []

    for i, r in enumerate(todo, 1):
        cmd = build_command(r, output_dir, args.passthrough)
        console_path = os.path.join(console_dir, r['tag'] + '.log')

        print('-' * 70)
        print(f"[{i}/{len(todo)}] {r['tag']}")
        print(f"  {CONFIGS[r['config']]['desc']}")
        print(f"  console -> {console_path}")
        sys.stdout.flush()

        status[r['tag']] = {
            'state': 'running',
            'dataset': r['dataset'],
            'config': r['config'],
            'seed': r['seed'],
            'command': ' '.join(cmd),
            'started': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        save_status(status_path, status)

        start = time.time()
        try:
            with open(console_path, 'w') as out:
                proc = subprocess.run(cmd, stdout=out,
                                      stderr=subprocess.STDOUT, check=False)
            returncode = proc.returncode
        except KeyboardInterrupt:
            status[r['tag']]['state'] = 'interrupted'
            save_status(status_path, status)
            print('\nInterrupted. Re-run this script to resume '
                  '(completed runs are skipped).')
            return
        except OSError as exc:
            returncode = -1
            print(f'  ERROR launching run: {exc}')

        duration = time.time() - start
        status[r['tag']].update({
            'state': 'completed' if returncode == 0 else 'failed',
            'returncode': returncode,
            'duration_seconds': round(duration, 1),
            'finished': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        save_status(status_path, status)

        if returncode == 0:
            print(f'  done in {format_duration(duration)}')
        else:
            failures.append((r['tag'], returncode, console_path))
            print(f'  FAILED (exit {returncode}) after {format_duration(duration)}')
            print(f'  see {console_path} for the traceback')
            if args.stop_on_failure:
                print('\nStopping because --stop-on-failure was set.')
                break

    print('\n' + '=' * 70)
    print(f'Suite finished in {format_duration(time.time() - suite_start)}')
    completed = sum(1 for v in status.values() if v.get('state') == 'completed')
    print(f'Completed runs on record: {completed}/{len(plan)}')
    if failures:
        print(f'\n{len(failures)} run(s) FAILED:')
        for tag, rc, path in failures:
            print(f'  {tag} (exit {rc}) -> {path}')
        print('\nFix the cause and re-run this script; successful runs are skipped.')

    run_aggregation(output_dir, args.metrics, paired=args.paired)


if __name__ == '__main__':
    main()
