# -*- coding: utf-8 -*-
"""Pre-run check for the D-MT4SR Office_Products handoff.

Run this ONCE before starting the real experiment suite. It verifies, in a
few minutes, everything that would otherwise only fail hours into an
unattended job:

  1. Python / PyTorch / CUDA versions and the GPU actually visible
  2. That every file the suite needs is present
  3. That the preprocessed dataset loads, with its user/item/relationship counts
  4. That a real training step runs, how long it takes, and its peak VRAM
  5. That BOTH configs in the study (baseline and v1) build and step
  6. A per-epoch time estimate, and the recommended chunk/batch size

It ends with a PASS or FAIL verdict. On PASS it prints the exact command to
run next. On FAIL it exits non-zero and says what to do about it.

    python preflight.py                       # the normal case
    python preflight.py --budget-hours=48     # also recommend an epoch cap

Nothing here writes into an experiment output directory, trains to
convergence, or changes any default. It is safe to run more than once.
"""
import argparse
import os
import platform
import sys
import time
import traceback

# Every failure path funnels through fail(), so the collaborator gets one
# actionable sentence instead of a bare traceback.
EXIT_OK = 0
EXIT_FAIL = 1

WIDTH = 74


def rule(char='-'):
    print(char * WIDTH, flush=True)


def section(title):
    print()
    rule('=')
    print(title)
    rule('=')


def item(label, value):
    print(f'  {label:<28} {value}', flush=True)


def fail(what, why, fix, exc=None):
    """Print an actionable failure and exit non-zero."""
    print()
    rule('=')
    print('PREFLIGHT FAILED')
    rule('=')
    print(f'  What failed : {what}')
    print(f'  Why         : {why}')
    print(f'  What to do  : {fix}')
    if exc is not None:
        print()
        print('  Full error follows, please include it if you report this:')
        print()
        traceback.print_exception(type(exc), exc, exc.__traceback__)
    print()
    print('Exiting with status 1. Do NOT start the experiment suite until this passes.')
    sys.exit(EXIT_FAIL)


def human_time(seconds):
    seconds = float(seconds)
    if seconds < 90:
        return f'{seconds:.1f}s'
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f'{h}h{m:02d}m'
    return f'{m}m{s:02d}s'


# torch.cuda.OutOfMemoryError only exists on newer PyTorch; fall back to the
# RuntimeError it subclasses so the OOM handler works on any version.
def _oom_error_types():
    try:
        import torch
        return (torch.cuda.OutOfMemoryError,)
    except (ImportError, AttributeError):
        return (RuntimeError,)


def recommend_batch_size(batch_small, peak_small, batch_large, peak_large,
                         vram_gb, floor_batch, headroom_frac=0.85):
    """Two-point fit of peak VRAM against batch size -> largest safe batch.

    peak(batch) ~= fixed + per_sample * batch. Measuring at two batch sizes
    separates the model/optimizer footprint (which does not scale with batch)
    from the activation and logits footprint (which does), so the
    extrapolation does not silently assume the whole measured peak is
    proportional to batch -- which would badly under-recommend on a large
    catalog, where the item embedding table alone is a fixed gigabyte or two.

    Returns (recommended, fixed_gib, per_sample_gib, raw_max) or None when the
    two measurements are too close to fit a line through.
    """
    span = batch_large - batch_small
    if span <= 0 or peak_large <= peak_small:
        return None
    per_sample = (peak_large - peak_small) / span
    fixed = peak_small - per_sample * batch_small
    if per_sample <= 0:
        return None
    raw_max = (headroom_frac * vram_gb - fixed) / per_sample
    if raw_max < floor_batch:
        # The card cannot even hold the batch we just measured; do not
        # recommend shrinking here -- the timed step above already proved
        # floor_batch runs, and OOM would have failed the run outright.
        return (floor_batch, fixed, per_sample, raw_max)
    # Round down to a multiple of 64 rather than to a power of two: powers of
    # two throw away up to half the card (a 246-batch capacity would round to
    # 128), and multiples of 64 are just as friendly to the tensor cores.
    recommended = max(int(raw_max) // 64 * 64, 64)
    return (max(recommended, floor_batch), fixed, per_sample, raw_max)


def largest_power_of_two_at_most(n):
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def main():
    ap = argparse.ArgumentParser(
        description='Pre-run environment and performance check for D-MT4SR.')
    ap.add_argument('--data_name', default='Office_Products',
                    help='dataset to check (default: Office_Products)')
    ap.add_argument('--batch_size', type=int, default=128,
                    help='batch size to time the reference steps at (default: 128, '
                         'the value every Appliances run used)')
    ap.add_argument('--steps', type=int, default=3,
                    help='timed training steps, after a warmup step (default: 3)')
    ap.add_argument('--eval-batches', type=int, default=3,
                    help='validation batches to time for the eval estimate (default: 3)')
    ap.add_argument('--budget-hours', type=float, default=None,
                    help='wall-clock hours available on the machine. If given, '
                         'preflight also prints the epoch cap that fits the budget.')
    ap.add_argument('--runs', type=int, default=10,
                    help='number of runs the budget must cover, for the projection '
                         '(default: 10 = 2 configs x 5 seeds)')
    ap.add_argument('--epochs', type=int, default=50,
                    help='epoch cap the run command will use, for the projection '
                         '(default: 50, matching README_HANDOFF.md)')
    ap.add_argument('--output-dir', default=None,
                    help='output directory to print in the suggested run command '
                         '(default: <dataset>_main/)')
    ap.add_argument('--no-batch-recommend', action='store_true',
                    help='skip the second measurement used to recommend a batch size')
    ap.add_argument('--allow-cpu', action='store_true',
                    help='run entirely on CPU and downgrade "no CUDA device" from a '
                         'failure to a warning. For testing this script off-GPU only '
                         '-- never for a real run. Note this also means preflight will '
                         'NOT touch the GPU, so it is safe to use while another job '
                         'is training.')
    args_cli = ap.parse_args()

    t_start = time.time()
    global _OOM_ERRORS
    warnings = []

    print()
    rule('=')
    print('D-MT4SR PREFLIGHT'.center(WIDTH))
    print(f'dataset: {args_cli.data_name}'.center(WIDTH))
    rule('=')

    # -----------------------------------------------------------------
    # 1. Environment
    # -----------------------------------------------------------------
    section('1. ENVIRONMENT')

    item('Host', platform.node())
    item('Platform', platform.platform())
    item('Python', sys.version.split()[0])
    item('Working directory', os.getcwd())

    try:
        import torch
    except ImportError as exc:
        fail('importing torch',
             'PyTorch is not installed in this Python environment.',
             'Install the CUDA build of PyTorch, e.g. '
             '"pip install torch --index-url https://download.pytorch.org/whl/cu121", '
             'then re-run this script.', exc)

    try:
        import numpy
        import scipy
    except ImportError as exc:
        fail('importing numpy/scipy',
             'A required package is missing.',
             'Run "pip install -r requirements.txt" and re-run this script.', exc)

    _OOM_ERRORS = _oom_error_types()

    item('PyTorch', torch.__version__)
    item('numpy', numpy.__version__)
    item('scipy', scipy.__version__)
    item('CUDA built with', torch.version.cuda or 'CPU-only build')
    item('cuDNN', torch.backends.cudnn.version() or 'n/a')

    cuda_ok = torch.cuda.is_available()
    item('CUDA available', cuda_ok)

    # --allow-cpu forces the CPU path even when a GPU is present, so that
    # testing this script never allocates VRAM out from under a job that is
    # already training on the card.
    if args_cli.allow_cpu and cuda_ok:
        print('  (--allow-cpu given: ignoring the visible GPU and running on CPU.)')
        cuda_ok = False

    if not cuda_ok:
        if not args_cli.allow_cpu:
            fail('CUDA availability check',
                 'torch.cuda.is_available() returned False, so training would '
                 'fall back to CPU and take weeks rather than hours.',
                 'Check that the GPU is visible ("nvidia-smi"), that you are on a '
                 'GPU node rather than the login node, and that the installed '
                 'PyTorch is a CUDA build (the "CUDA built with" line above must '
                 'not say CPU-only). Then re-run this script.')
        print()
        print('  !! WARNING: running WITHOUT CUDA because --allow-cpu was given.')
        print('  !! Timings below are meaningless for the real job.')
        vram_gb = None
        device = torch.device('cpu')
    else:
        item('GPU count', torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            item(f'  device {i}', f'{props.name}  '
                                  f'{props.total_memory / (1024 ** 3):.1f} GiB  '
                                  f'sm_{props.major}{props.minor}')
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024 ** 3)
        device = torch.device('cuda')

    # -----------------------------------------------------------------
    # 2. Files
    # -----------------------------------------------------------------
    section('2. REQUIRED FILES')

    needed_modules = ['main.py', 'trainers.py', 'modules.py', 'seqmodels.py',
                      'datasets.py', 'utils.py', 'run_experiments.py',
                      'aggregate_results.py']
    missing = [f for f in needed_modules if not os.path.exists(f)]
    for f in needed_modules:
        item(f, 'found' if os.path.exists(f) else 'MISSING')

    if missing:
        fail('the required-files check',
             f'These files are missing from the working directory: '
             f'{", ".join(missing)}.',
             'You are probably running from the wrong directory. cd into the '
             'folder that contains main.py (the same folder as this script) and '
             're-run. If the files really are absent, the copy is incomplete -- '
             'ask for the package again.')

    data_path = os.path.join('data', f'{args_cli.data_name}Partitioned_5core.npy')
    exists = os.path.exists(data_path)
    item(data_path, f'found ({os.path.getsize(data_path) / 1e6:.0f} MB)'
         if exists else 'MISSING')
    if not exists:
        fail('the dataset check',
             f'{data_path} does not exist.',
             'The preprocessed dataset must sit at data/'
             f'{args_cli.data_name}Partitioned_5core.npy relative to the current '
             'directory -- utils.get_user_seqs_MoHRdata() hardcodes that path, so '
             'it cannot be elsewhere. Check the file finished copying '
             '(Office_Products is ~142 MB).')

    # -----------------------------------------------------------------
    # 3. Dataset
    # -----------------------------------------------------------------
    section('3. DATASET')

    try:
        from utils import (get_user_seqs_MoHRdata, set_seed,
                           detect_total_vram_gb, chunk_size_for,
                           predicted_chunk_peak_gib, CPU_RAM_BUDGET_GB)
        t0 = time.time()
        (user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users,
         user_seq_mask_mat_rel, relationships_ind_map, Item,
         user_seq_times) = get_user_seqs_MoHRdata(args_cli.data_name)
        load_seconds = time.time() - t0
    except Exception as exc:
        fail('loading the preprocessed dataset',
             f'{data_path} exists but could not be loaded or unpacked.',
             'The file is most likely truncated or corrupted in transfer -- '
             'check its size against the sender\'s copy and re-copy it. '
             '(Office_Products should be ~142 MB.)', exc)

    item('Load time', human_time(load_seconds))
    item('Users', f'{num_users:,}')
    item('Items (max index)', f'{max_item:,}')
    item('item_size', f'{max_item + 2:,}  (+1 padding, +1 mask)')
    item('Relationship types', f'{len(relationships_ind_map)}  '
                               f'({", ".join(map(str, relationships_ind_map))})')
    item('Items with relations', f'{len(Item):,}')
    item('Real timestamps', user_seq_times is not None)
    item('Mean sequence length',
         f'{sum(len(s) for s in user_seq) / max(len(user_seq), 1):.1f}')

    # -----------------------------------------------------------------
    # 4. Chunk size
    # -----------------------------------------------------------------
    section('4. RELATION-LOSS CHUNK SIZE')

    item_size = max_item + 2
    num_rel = len(relationships_ind_map)
    if vram_gb is None:
        auto_chunk = chunk_size_for(CPU_RAM_BUDGET_GB, item_size,
                                    args_cli.batch_size, 100, num_rel)
        print(f'  No GPU detected; sizing against an assumed '
              f'{CPU_RAM_BUDGET_GB:.0f} GiB of host RAM -> {auto_chunk}.')
    else:
        # Must match utils.resolve_rel_loss_chunk_size() exactly, or preflight
        # advertises one chunk size and the run uses another.
        auto_chunk = chunk_size_for(vram_gb, item_size,
                                    args_cli.batch_size, 100, num_rel)

    # The tensor the whole memory story is about.
    logits_gb = (args_cli.batch_size * 100 * num_rel * item_size * 4) / (1024 ** 3)
    item('Full logits tensor',
         f'{args_cli.batch_size} x 100 x {num_rel} x {item_size:,} float32 '
         f'= {logits_gb:.1f} GiB (forward only)')
    item('Detected VRAM', f'{vram_gb:.1f} GiB' if vram_gb else 'none')
    item('--rel_loss_chunk_size=auto',
         '0 (unchunked -- the full tensor fits)' if auto_chunk == 0 else str(auto_chunk))
    predicted_peak = predicted_chunk_peak_gib(auto_chunk, item_size,
                                              args_cli.batch_size, 100, num_rel)
    item('Predicted peak for that',
         f'{predicted_peak:.1f} GiB'
         + (f'  ({100 * predicted_peak / vram_gb:.0f}% of VRAM)' if vram_gb else ''))
    print()
    print('  Chunking is mathematically neutral: same loss value, same gradients,')
    print('  bounded peak memory, some recomputation cost. It cannot change results.')

    # -----------------------------------------------------------------
    # 5. Timed training steps
    # -----------------------------------------------------------------
    section('5. TIMED TRAINING STEPS')

    import types
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

    def build_args(model_name, batch_size, chunk_size):
        """An args namespace matching what run_experiments.py passes to main.py."""
        a = types.SimpleNamespace(
            data_dir='../data/', output_dir='preflight_scratch/',
            data_name=args_cli.data_name, do_eval=False, ckp=10,
            model_name=model_name,
            hidden_size=128, num_hidden_layers=1, num_attention_heads=1,
            hidden_act='gelu', attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.3, initializer_range=0.02,
            max_seq_length=100, distance_metric='wasserstein', pvn_weight=0.1,
            rel_loss_weight=0.1, outseq_rel_loss_weight=0.05,
            use_time_decay=False, dynamic_loss_weights=False,
            popularity_neg_sampling=False, time_scale=86400.0,
            time_decay_floor=0.1, gate_residual=False, gate_scale_init=0.0,
            gate_pairwise=False, gate_use_rel_mask=False, gate_per_head=False,
            gate_temperature=1.0, gate_entropy_weight=0.0,
            gate_entropy_epochs=0, gate_lr_scale=1.0, ema_decay=0.0,
            val_smooth_window=1, patience=50, popneg_mix=1.0,
            time_decay_log=False, rel_score_norm='none',
            rel_loss_chunk_size=chunk_size,
            lr=0.001, batch_size=batch_size, epochs=100,
            no_cuda=(vram_gb is None), log_freq=1, seed=42,
            weight_decay=0.0, adam_beta1=0.9, adam_beta2=0.999, gpu_id='0',
        )
        a.cuda_condition = cuda_ok and not a.no_cuda
        a.item_size = item_size
        a.num_users = num_users
        a.mask_id = max_item + 1
        a.has_real_timestamps = user_seq_times is not None
        a.train_matrix = valid_rating_matrix
        a.log_file = os.devnull
        a.checkpoint_path = os.devnull
        return a

    def build_stack(model_name, batch_size, chunk_size):
        from datasets import (RelationAwareSASRecDataset,
                              DynamicRelationAwareSASRecDataset)
        from seqmodels import (RelationAwareSASRecModel,
                               DynamicRelationAwareSASRecModel)
        from trainers import (RelationAwareSASRecModelTrainer,
                              DynamicRelationAwareSASRecModelTrainer)
        a = build_args(model_name, batch_size, chunk_size)
        set_seed(a.seed)
        dynamic = model_name == 'DynamicRelationAwareSASRecModel'
        if dynamic:
            mk = lambda dt: DynamicRelationAwareSASRecDataset(
                a, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item,
                data_type=dt, sampling_probs=None, user_seq_times=user_seq_times)
        else:
            mk = lambda dt: RelationAwareSASRecDataset(
                a, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item,
                data_type=dt)
        train_ds, eval_ds = mk('train'), mk('valid')
        train_dl = DataLoader(train_ds, sampler=RandomSampler(train_ds),
                              batch_size=batch_size)
        eval_dl = DataLoader(eval_ds, sampler=SequentialSampler(eval_ds),
                             batch_size=batch_size)
        model_cls = (DynamicRelationAwareSASRecModel if dynamic
                     else RelationAwareSASRecModel)
        trainer_cls = (DynamicRelationAwareSASRecModelTrainer if dynamic
                       else RelationAwareSASRecModelTrainer)
        model = model_cls(a, relationships_ind_map)
        trainer = trainer_cls(model, train_dl, eval_dl, eval_dl, a)
        return a, trainer, train_dl, eval_dl

    def one_step(trainer, batch):
        """Exactly the body of the trainer's iteration() training branch, for
        one batch.

        The two configs feed different batches: the dynamic dataset appends a
        9th tensor of per-interaction timestamps and passes it to finetune().
        Branching on the batch width keeps this faithful to both trainers
        rather than to whichever one happened to be written first.
        """
        batch = tuple(t.to(trainer.device) for t in batch)
        if len(batch) == 9:
            (_, input_ids, target_pos, target_neg, _, rel_seq_masks,
             item_rel, item_rel_pos, input_times) = batch
            sequence_output, sequence_input, rel_embs, _ = trainer.model.finetune(
                input_ids, rel_seq_masks[:, :, :-1, :-1], input_times=input_times)
        else:
            (_, input_ids, target_pos, target_neg, _, rel_seq_masks,
             item_rel, item_rel_pos) = batch
            sequence_output, sequence_input, rel_embs, _ = trainer.model.finetune(
                input_ids, rel_seq_masks[:, :, :-1, :-1])
        pred_loss, _ = trainer.pred_loss(sequence_output, target_pos, target_neg)
        intra = trainer.relation_loss(sequence_input, sequence_output, target_pos,
                                      target_neg, rel_seq_masks, rel_embs[-1])
        inter = trainer.relation_outside_seq_loss(item_rel, item_rel_pos, rel_embs[-1])
        alpha, beta = trainer.get_loss_weights()
        loss = pred_loss + alpha * intra + beta * inter
        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()
        trainer._post_step()
        return float(loss.item())

    def time_steps(model_name, batch_size, chunk_size, n_steps, label):
        """Warmup step + n timed steps. Returns (mean_seconds, peak_gib, steps_per_epoch)."""
        if cuda_ok and vram_gb is not None:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        a, trainer, train_dl, eval_dl = build_stack(model_name, batch_size, chunk_size)
        # The real training loop enables anomaly detection, which roughly
        # doubles backward cost. Mirror it so these timings match the real job
        # instead of flattering it.
        torch.autograd.set_detect_anomaly(True)
        trainer.model.train()
        steps_per_epoch = len(train_dl)
        it = iter(train_dl)
        times = []
        try:
            for i in range(n_steps + 1):          # step 0 is warmup, not timed
                batch = next(it)
                if cuda_ok and vram_gb is not None:
                    torch.cuda.synchronize()
                t0 = time.time()
                loss_value = one_step(trainer, batch)
                if cuda_ok and vram_gb is not None:
                    torch.cuda.synchronize()
                dt = time.time() - t0
                if i == 0:
                    print(f'  [{label}] warmup step: {dt:.2f}s, loss {loss_value:.4f}')
                else:
                    times.append(dt)
                    print(f'  [{label}] step {i}: {dt:.2f}s, loss {loss_value:.4f}',
                          flush=True)
        except _OOM_ERRORS as exc:
            fail(f'a training step at batch_size={batch_size}',
                 'The GPU ran out of memory during a real training step.',
                 f'Re-run preflight with a smaller batch and an explicit chunk '
                 f'size, e.g. "python preflight.py --batch_size=64". If that also '
                 f'OOMs, another process is probably holding the GPU -- check '
                 f'"nvidia-smi". Detected VRAM was '
                 f'{vram_gb:.1f} GiB and the chunk size in use was {chunk_size}.',
                 exc)
        except StopIteration:
            fail('iterating the training data',
                 'The training dataloader ran out of batches immediately, which '
                 'means the dataset loaded but is empty.',
                 'The preprocessed .npy is present but does not contain usable '
                 'sequences. Re-copy it from the sender.')
        except Exception as exc:
            fail(f'a training step ({model_name})',
                 'A real training step raised an exception.',
                 'This is a code or environment problem, not a capacity one. '
                 'Send the traceback below back to the sender -- do not start '
                 'the suite.', exc)
        finally:
            torch.autograd.set_detect_anomaly(False)

        peak = (torch.cuda.max_memory_allocated() / (1024 ** 3)
                if (cuda_ok and vram_gb is not None) else float('nan'))
        mean = sum(times) / len(times)
        del trainer
        return mean, peak, steps_per_epoch, eval_dl

    print(f'  config      : baseline (RelationAwareSASRecModel, = MT4SR)')
    print(f'  batch size  : {args_cli.batch_size}')
    print(f'  chunk size  : {auto_chunk} '
          f'{"(unchunked)" if auto_chunk == 0 else ""}')
    print()

    mean_step, peak_gib, steps_per_epoch, eval_dl = time_steps(
        'RelationAwareSASRecModel', args_cli.batch_size, auto_chunk,
        args_cli.steps, 'baseline')

    print()
    item('Mean step time', f'{mean_step:.2f}s')
    item('Steps per epoch', f'{steps_per_epoch:,}')
    if vram_gb is not None:
        used_frac = peak_gib / vram_gb
        item('Peak VRAM (allocated)', f'{peak_gib:.1f} GiB of {vram_gb:.1f} GiB '
                                      f'({100 * used_frac:.0f}%)')
        item('Predicted vs measured',
             f'{predicted_peak:.1f} GiB predicted, {peak_gib:.1f} GiB measured')
        if used_frac > 0.85:
            over_vram = peak_gib > vram_gb
            smaller = max(auto_chunk // 2, 1024) if auto_chunk else 16384
            warnings.append("peak VRAM too close to this card's limit")
            print()
            print('  ' + '!' * (WIDTH - 4))
            print("  !! PEAK MEMORY IS AT OR OVER THIS CARD'S LIMIT")
            print('  ' + '!' * (WIDTH - 4))
            if over_vram:
                print(f'  Measured peak ({peak_gib:.1f} GiB) EXCEEDS total VRAM '
                      f'({vram_gb:.1f} GiB).')
                print('  On Linux this is an out-of-memory crash. On Windows the driver')
                print('  silently spills into system RAM instead, which keeps running but')
                print('  is many times slower -- and is very likely why this card is slow.')
            else:
                print(f'  Measured peak is {100 * used_frac:.0f}% of VRAM, leaving almost no')
                print('  headroom for allocator fragmentation. An OOM mid-run is likely.')
            print()
            print(f'  Fix: append --rel_loss_chunk_size={smaller} after --passthrough.')
            print('  That halves the peak and is mathematically identical -- same loss,')
            print('  same gradients. Then re-run this preflight to confirm.')
            print('  ' + '!' * (WIDTH - 4))

    train_epoch_seconds = mean_step * steps_per_epoch
    item('Estimated train time/epoch', human_time(train_epoch_seconds))

    # -----------------------------------------------------------------
    # 6. Validation cost
    # -----------------------------------------------------------------
    section('6. VALIDATION COST')
    print('  Every epoch is followed by a full-sort evaluation over all users,')
    print('  which on a large catalog is a real fraction of the epoch cost.')
    print()

    eval_epoch_seconds = 0.0
    try:
        import numpy as np
        _, trainer_e, _, eval_dl_e = build_stack(
            'RelationAwareSASRecModel', args_cli.batch_size, auto_chunk)
        trainer_e.model.eval()
        eval_batches = len(eval_dl_e)
        it = iter(eval_dl_e)
        t0 = time.time()
        counted = 0
        with torch.no_grad():
            for _ in range(args_cli.eval_batches):
                batch = next(it)
                batch = tuple(t.to(trainer_e.device) for t in batch)
                user_ids, input_ids, rel_seq_masks = batch[0], batch[1], batch[5]
                out, _, _, _ = trainer_e.model.finetune(
                    input_ids, rel_seq_masks[:, :, :-1, :-1])
                rating_pred = trainer_e.relation_predict_full(out[:, -1, :])
                rating_pred = rating_pred.cpu().data.numpy().copy()
                bidx = user_ids.cpu().numpy()
                rating_pred[trainer_e.args.train_matrix[bidx].toarray() > 0] = 0
                np.argpartition(rating_pred, -40)[:, -40:]
                counted += 1
        per_eval_batch = (time.time() - t0) / max(counted, 1)
        eval_epoch_seconds = per_eval_batch * eval_batches
        item('Validation batches', f'{eval_batches:,}')
        item('Time per eval batch', f'{per_eval_batch:.2f}s')
        item('Estimated eval time/epoch', human_time(eval_epoch_seconds))
        del trainer_e
    except Exception as exc:
        print('  Could not time validation; the per-epoch estimate below covers')
        print(f'  training only and is therefore optimistic. ({type(exc).__name__}: {exc})')

    epoch_seconds = train_epoch_seconds + eval_epoch_seconds

    # -----------------------------------------------------------------
    # 7. Both configs build
    # -----------------------------------------------------------------
    section('7. BOTH STUDY CONFIGS BUILD AND STEP')
    print('  The suite runs two configs. A config that only fails when its turn')
    print('  comes would waste every hour spent on the other one, so both are')
    print('  smoke-tested here.')
    print()
    v1_step, v1_peak, _, _ = time_steps(
        'DynamicRelationAwareSASRecModel', args_cli.batch_size, auto_chunk, 1, 'v1')
    item('v1 step time', f'{v1_step:.2f}s')
    if vram_gb is not None:
        item('v1 peak VRAM', f'{v1_peak:.1f} GiB')

    # -----------------------------------------------------------------
    # 8. Batch size recommendation
    # -----------------------------------------------------------------
    section('8. RECOMMENDED BATCH SIZE')

    recommended_batch = args_cli.batch_size
    max_batch = None
    if vram_gb is None or args_cli.no_batch_recommend:
        print('  Skipped (no GPU, or --no-batch-recommend given).')
    else:
        # Two-point fit: peak(batch) ~= fixed + per_sample * batch. Measuring at
        # two batch sizes separates the model/optimizer footprint (which does
        # not scale) from the activation and logits footprint (which does).
        probe_batch = max(16, args_cli.batch_size // 2)
        print(f'  Measuring peak VRAM at batch {probe_batch} as a second point,')
        print(f'  to separate the fixed footprint from the per-sample cost.')
        print()
        small_step, small_peak, _, _ = time_steps(
            'RelationAwareSASRecModel', probe_batch, auto_chunk, 1, f'batch{probe_batch}')
        fit = recommend_batch_size(probe_batch, small_peak,
                                   args_cli.batch_size, peak_gib,
                                   vram_gb, args_cli.batch_size)
        if fit is None:
            print('  Measurements too close to extrapolate from; keeping '
                  f'batch {args_cli.batch_size}.')
        else:
            fitted_batch, fixed, per_sample, raw_max = fit
            print()
            item('Fixed footprint', f'{fixed:.2f} GiB')
            item('Per-sample cost', f'{per_sample * 1024:.1f} MiB/sample')
            item('Usable (85% of VRAM)', f'{0.85 * vram_gb:.1f} GiB')
            item('Unrounded capacity', f'{int(raw_max)}')
            item('Largest batch that fits', f'{fitted_batch} (rounded to a multiple of 64)')
            fit_steps = -(-num_users // fitted_batch)
            item('...which would give', f'{fit_steps:,} optimizer steps per epoch '
                                        f'(vs {steps_per_epoch:,} at batch {args_cli.batch_size})')
            if fit_steps < 100:
                print()
                print(f'  That is only {fit_steps} weight updates per epoch. Memory is not the')
                print('  binding constraint at this catalog size -- the chunked loss keeps peak')
                print('  memory flat as the batch grows -- so this number says what FITS, not')
                print('  what is sensible. Treat it as an upper bound, not a suggestion.')
            max_batch = fitted_batch
        print()
        print('  READ THIS BEFORE CHANGING THE BATCH SIZE')
        print('  ' + '-' * (WIDTH - 4))
        print('  Batch size is the only tunable here that changes results, for two')
        print('  reasons: it changes the gradient noise scale, and at a fixed epoch')
        print('  budget it changes the NUMBER OF OPTIMIZER STEPS. Doubling the batch')
        print('  halves the updates per epoch at the same learning rate, which can')
        print('  leave a 100-epoch run under-trained -- weakening both arms of the')
        print('  comparison and making a real difference harder to detect.')
        print()
        if auto_chunk == 0:
            print('  The free speedup on this card is NOT a bigger batch: chunking is')
            print('  already off here (auto resolved to 0), so there is no recomputation')
            print('  left to remove. Note that unchunked peak memory scales WITH batch')
            print('  size, so doubling the batch doubles a ~39 GiB tensor.')
        else:
            print(f'  Note that chunking is ON here (auto resolved to {auto_chunk}), so a')
            print('  larger batch also raises the number of chunks and buys less speed')
            print('  than it appears to.')
        print()
        print('  Both configs must use the SAME batch size or the paired-seed comparison')
        print('  is meaningless. The Appliances runs used 128, so the command below keeps')
        print('  128. Change it only deliberately, and for both configs at once.')

    # -----------------------------------------------------------------
    # 9. Projection
    # -----------------------------------------------------------------
    section('9. RUNTIME PROJECTION')

    item('Time per epoch', human_time(epoch_seconds))
    item('Epochs planned', args_cli.epochs)
    per_run = epoch_seconds * args_cli.epochs
    item('Time per run', human_time(per_run))
    item('Runs planned', f'{args_cli.runs} (2 configs x {args_cli.runs // 2} seeds)')
    item('Total (upper bound)', human_time(per_run * args_cli.runs))
    print()
    print('  Early stopping (patience 50) can end a run sooner -- though at a 50-epoch')
    print('  cap it will not fire, so its role here is just to keep the best-scoring')
    print('  checkpoint. Treat the figure above as a LOWER bound: it times a few steps')
    print('  early in epoch 0 and excludes checkpoint saves and dataloader warmup, and')
    print('  on the reference card the measured epoch ran above the estimate. Budget')
    print('  with headroom rather than to this number exactly.')
    if eval_epoch_seconds > train_epoch_seconds:
        print()
        print(f'  NOTE: validation is the larger half of the epoch here '
              f'({100 * eval_epoch_seconds / epoch_seconds:.0f}%).')
        print('  That part is mostly CPU work (host-side masking and partial sort over')
        print('  the full catalog), so a faster GPU may not shrink it. If this run is')
        print('  slower than expected, the CPU is the more likely cause than the GPU.')

    if args_cli.budget_hours:
        budget_seconds = args_cli.budget_hours * 3600
        per_run_budget = budget_seconds / args_cli.runs
        fits = int(per_run_budget // epoch_seconds) if epoch_seconds > 0 else 0
        print()
        item('Budget', f'{args_cli.budget_hours:g} hours')
        item('Budget per run', human_time(per_run_budget))
        item('Epochs that fit', fits)
        if fits < args_cli.epochs:
            print()
            print(f'  !! {args_cli.epochs} epochs x {args_cli.runs} runs does NOT fit '
                  f'in {args_cli.budget_hours:g} hours.')
            print(f'  !! Either pass --epochs={max(fits, 1)} to the run command, or run')
            print(f'  !! the reduced-seed fallback command from README_HANDOFF.md.')

    # -----------------------------------------------------------------
    # 10. Verdict
    # -----------------------------------------------------------------
    section('VERDICT')

    if warnings:
        print('  PASS WITH WARNINGS -- it runs, but read the flagged section above.')
        for w in warnings:
            print(f'    - {w}')
    else:
        print('  PASS -- the environment, the data, and both configs are working.')
    print()
    item('Recommended chunk size',
         'auto -> 0 (unchunked)' if auto_chunk == 0 else f'auto -> {auto_chunk}')
    item('Batch size in the command', f'{recommended_batch}  (matches the Appliances runs)')
    if max_batch and max_batch > recommended_batch:
        item('Largest batch that fits', f'{max_batch}  (opt-in, see section 8)')
    item('Preflight took', human_time(time.time() - t_start))
    print()
    print('  Next, run this exact command from this directory:')
    print()
    seeds = '1 7 42 123 456'
    out_dir = args_cli.output_dir or f'{args_cli.data_name.split("_")[0].lower()}_main/'
    print(f'    python run_experiments.py \\')
    print(f'      --datasets {args_cli.data_name} \\')
    print(f'      --configs baseline v1 \\')
    print(f'      --seeds {seeds} \\')
    print(f'      --output-dir {out_dir} \\')
    print(f'      --passthrough --epochs={args_cli.epochs} '
          f'--batch_size={recommended_batch}')
    print()
    print('  (On Windows, put it on one line and drop the backslashes.)')
    print()
    print('  If several GPUs are free, README_HANDOFF.md section 4 Option B splits')
    print('  the seeds across them and finishes in roughly the time of one shard.')
    print()
    rule('=')
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
