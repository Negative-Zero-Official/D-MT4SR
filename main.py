# -*- coding: utf-8 -*-
# @Time    : 2020/4/25 22:59

import os
import time
import numpy as np
import random
import torch
import argparse

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from datasets import SASRecDataset, RelationAwareSASRecDataset, DynamicRelationAwareSASRecDataset
from trainers import FinetuneTrainer, DistSAModelTrainer, RelationAwareSASRecModelTrainer, DynamicRelationAwareSASRecModelTrainer
from seqmodels import SASRecModel, DistSAModel, DistMeanSAModel, RelationAwareSASRecModel, DynamicRelationAwareSASRecModel
from utils import EarlyStopping, get_user_seqs, get_item2attribute_json, check_path, set_seed, get_user_seqs_MoHRdata, \
        compute_item_popularity, build_popularity_sampling_probs, \
        rel_loss_chunk_size_arg, resolve_rel_loss_chunk_size


# D-MT4SR v2 options are appended to the run name ONLY when they differ from
# the v1 defaults. That keeps every filename produced by the previous version
# of this script byte-identical, so already-collected baseline/v1/dynloss logs
# still group correctly in aggregate_results.py instead of splitting off into
# new configs the moment these flags exist.
_V2_NAME_PARTS = [
    ('gate_residual', False, lambda v: 'gres'),
    ('gate_scale_init', 0.0, lambda v: f'gsi{v}'),
    ('gate_pairwise', False, lambda v: 'gpair'),
    ('gate_use_rel_mask', False, lambda v: 'gmask'),
    ('gate_per_head', False, lambda v: 'ghead'),
    ('gate_temperature', 1.0, lambda v: f'gtemp{v}'),
    ('gate_entropy_weight', 0.0, lambda v: f'gent{v}'),
    ('gate_entropy_epochs', 0, lambda v: f'gente{v}'),
    ('gate_lr_scale', 1.0, lambda v: f'glr{v}'),
]
# These apply to the MT4SR baseline too, so they're outside the model check.
_STABILITY_NAME_PARTS = [
    ('rel_score_norm', 'none', lambda v: f'rsn-{v}'),
    ('ema_decay', 0.0, lambda v: f'ema{v}'),
    ('val_smooth_window', 1, lambda v: f'vsm{v}'),
    ('patience', 50, lambda v: f'pat{v}'),
]


def _suffix_from(args, parts):
    out = ''
    for attr, default, fmt in parts:
        value = getattr(args, attr, default)
        if value != default:
            out += '-' + fmt(value)
    return out


def _append_variant_suffixes(args, args_str):
    if args.model_name == 'DynamicRelationAwareSASRecModel':
        args_str += _suffix_from(args, _V2_NAME_PARTS)
    return args_str + _suffix_from(args, _STABILITY_NAME_PARTS)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', default='../data/', type=str)
    parser.add_argument('--output_dir', default='output/', type=str)
    parser.add_argument('--data_name', default='Beauty', type=str)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--ckp', default=10, type=int, help="pretrain epochs 10, 20, 30...")

    # model args
    parser.add_argument("--model_name", default='Finetune_full', type=str)
    parser.add_argument("--hidden_size", type=int, default=64, help="hidden size of transformer model")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument('--num_attention_heads', default=2, type=int)
    parser.add_argument('--hidden_act', default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument('--max_seq_length', default=50, type=int)
    parser.add_argument('--distance_metric', default='wasserstein', type=str)
    parser.add_argument('--pvn_weight', default=0.1, type=float)
    parser.add_argument('--rel_loss_weight', default=0.0, type=float)
    parser.add_argument('--outseq_rel_loss_weight', default=0.0, type=float)

    # D-MT4SR args (model_name='DynamicRelationAwareSASRecModel'). All default to False/off,
    # so DynamicRelationAwareSASRecModel with no flags set differs from MT4SR's
    # RelationAwareSASRecModel only in the dynamic (context-conditioned) relation
    # weighting baked into the encoder -- everything below is opt-in on top of that.
    parser.add_argument('--use_time_decay', action='store_true',
                        help="D-MT4SR: modulate relation attention weights by a learnable "
                             "per-relationship decay over sequence-position distance")
    parser.add_argument('--dynamic_loss_weights', action='store_true',
                        help="D-MT4SR: learn a multiplicative correction on top of "
                             "rel_loss_weight/outseq_rel_loss_weight instead of using them as fixed constants")
    parser.add_argument('--popularity_neg_sampling', action='store_true',
                        help="D-MT4SR: use popularity-aware (freq^0.75) negative sampling "
                             "instead of uniform random sampling")
    parser.add_argument('--time_scale', default=86400.0, type=float,
                        help="D-MT4SR: normalization divisor applied to raw timestamp gaps "
                             "(only used when --use_time_decay and real timestamps are available; "
                             "default 86400 = seconds/day)")
    parser.add_argument('--time_decay_floor', default=0.1, type=float,
                        help="D-MT4SR: minimum fraction of the relation attention signal that "
                             "survives even for maximally distant/old item pairs (only used with "
                             "--use_time_decay). Prevents the learned decay from fully collapsing "
                             "the relation signal on sparse datasets where interactions are far "
                             "apart in time or position. 0.0 disables the floor (original behavior).")

    # --- D-MT4SR v2: relation-gate options -------------------------------
    # Everything here defaults to the v1 behavior, so existing runs and log
    # filenames are unaffected unless a flag is explicitly set.
    parser.add_argument('--gate_residual', action='store_true',
                        help="D-MT4SR v2: express the context-conditioned relation gate as a "
                             "scaled perturbation of MT4SR's static relationship_weights, "
                             "instead of replacing them. With gate_scale initialized to 0 the "
                             "model starts *exactly* at MT4SR, making D-MT4SR a strict "
                             "generalization of the baseline and removing the random relation "
                             "prior responsible for most of the seed-to-seed variance.")
    parser.add_argument('--gate_scale_init', default=0.0, type=float,
                        help="Initial value of the learnable gate_scale (only used with "
                             "--gate_residual). 0.0 = start at the MT4SR solution.")
    parser.add_argument('--gate_pairwise', action='store_true',
                        help="D-MT4SR v2: condition the relation gate on the (query, key) PAIR "
                             "via a low-rank additive term, not just the query position -- "
                             "which relationship matters is naturally a property of the pair.")
    parser.add_argument('--gate_use_rel_mask', action='store_true',
                        help="D-MT4SR v2: add a learnable per-relationship logit bonus at the "
                             "pairs where that relationship is actually observed in the item "
                             "graph. These masks are computed by the dataset and never "
                             "consulted by MT4SR's attention, so this is otherwise-discarded signal.")
    parser.add_argument('--gate_per_head', action='store_true',
                        help="D-MT4SR v2: give each attention head its own relation "
                             "distribution instead of sharing one across heads. "
                             "Only meaningful with --num_attention_heads > 1.")
    parser.add_argument('--gate_temperature', default=1.0, type=float,
                        help="D-MT4SR v2: softmax temperature for the relation gate. "
                             ">1 flattens the distribution, <1 sharpens it. 1.0 = off.")
    parser.add_argument('--gate_entropy_weight', default=0.0, type=float,
                        help="D-MT4SR v2: reward (subtract from the loss) the entropy of the "
                             "relation distribution, discouraging early collapse onto a single "
                             "relationship. 0.0 = off. Try 0.01-0.1.")
    parser.add_argument('--gate_entropy_epochs', default=0, type=int,
                        help="Anneal --gate_entropy_weight linearly to 0 over this many epochs. "
                             "0 = constant weight for the whole run.")
    parser.add_argument('--gate_lr_scale', default=1.0, type=float,
                        help="D-MT4SR v2: learning-rate multiplier for the gate parameters only "
                             "(relation_gate, gate_scale, relation_mask_bias). 1.0 = off. "
                             "Try 0.1 -- the gate is a small auxiliary module on an otherwise "
                             "tuned architecture and does not need to move as fast.")

    # --- Stability / model-selection options (apply to BOTH baseline and D-MT4SR) ---
    parser.add_argument('--ema_decay', default=0.0, type=float,
                        help="Keep an exponential moving average of the weights and use it for "
                             "validation and checkpointing. 0.0 = off (original behavior). "
                             "Try 0.999. Reduces the 'lucky epoch' component of model selection.")
    parser.add_argument('--val_smooth_window', default=1, type=int,
                        help="Average validation MRR over this many recent epochs before "
                             "comparing for early stopping / checkpointing. 1 = original "
                             "behavior. Try 3 on small datasets where per-epoch validation is noisy.")
    parser.add_argument('--patience', default=50, type=int,
                        help="Early-stopping patience in epochs (original MT4SR setup: 50).")

    parser.add_argument('--popneg_mix', default=1.0, type=float,
                        help="Probability that a negative is drawn from the popularity "
                             "distribution rather than uniformly (only used with "
                             "--popularity_neg_sampling). 1.0 = all-popularity (original). "
                             "Try 0.5 -- pure popularity sampling systematically suppresses "
                             "exactly the items full-sort metrics reward ranking highly.")
    parser.add_argument('--time_decay_log', action='store_true',
                        help="D-MT4SR: apply log1p compression to timestamp gaps before the "
                             "decay. Amazon data spans ~20 years, so raw day-scale gaps saturate "
                             "the decay for essentially every distant pair; log gaps keep it "
                             "discriminative between a 1-week and a 1-year gap.")

    parser.add_argument('--rel_score_norm', default='none',
                        choices=['none', 'std', 'layernorm'],
                        help="Normalize the relation attention scores before adding them to "
                             "the ordinary attention logits. MEASURED: at hidden_size=128 the "
                             "relation scores are ~4 orders of magnitude larger than the "
                             "ordinary attention scores (~45000 vs ~1.7), which saturates the "
                             "attention softmax to near one-hot (entropy 0.002 out of a "
                             "possible 4.61) and keeps it there through training. 'std' "
                             "standardizes them per query row and restores a healthy attention "
                             "distribution. Available for the MT4SR baseline too, so it can be "
                             "run as a control. 'none' = original behavior.")

    parser.add_argument('--rel_loss_chunk_size', default=0, type=rel_loss_chunk_size_arg,
                        help="Compute the inter-sequence relation loss in row-chunks of this "
                             "size under gradient checkpointing, bounding peak GPU memory. "
                             "Mathematically identical to the unchunked loss (same value and "
                             "gradients), just slower. 0 = disabled (original behavior). "
                             "Needed for large-catalog datasets such as Office_Products, where "
                             "the full (batch*seq*num_rel, item_size) logits tensor can exceed "
                             "GPU memory. Try 2048 or 1024 if you hit CUDA OOM. "
                             "Pass 'auto' to pick a value from the detected VRAM "
                             "(>=70 GiB unchunked, 35-70 -> 32768, 20-35 -> 16384, else 8192); "
                             "the chosen value is printed and appears in the log's args line, "
                             "so an auto run stays as reproducible as an explicit one.")

    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--batch_size", type=int, default=128, help="number of batch_size")
    parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--gpu_id", type=str, default="0", help="gpu_id")

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)


    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # '--rel_loss_chunk_size=auto' is resolved to a concrete integer here,
    # before args is printed or written to the log, so the log line records the
    # value that was actually used rather than the sentinel. An explicitly
    # given value passes through untouched. Chunking is mathematically neutral
    # (same loss, same gradients), so this only affects peak memory and speed,
    # never the numbers a run produces.
    resolve_rel_loss_chunk_size(args)

    args.data_file = args.data_dir + args.data_name + '.txt'
    #item2attribute_file = args.data_dir + args.data_name + '_item2attributes.json'

    #user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users = \
    #    get_user_seqs(args.data_file)
    user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users, user_seq_mask_mat_rel, relationships_ind_map, Item, user_seq_times = \
            get_user_seqs_MoHRdata(args.data_name)

    #item2attribute, attribute_size = get_item2attribute_json(item2attribute_file)

    args.item_size = max_item + 2
    args.num_users = num_users
    args.mask_id = max_item + 1
    #args.attribute_size = attribute_size + 1
    # D-MT4SR: True only if the preprocessed .npy actually carries real
    # per-interaction timestamps (see preprocess_fromscratch.py /
    # utils.get_user_seqs_MoHRdata). Older preprocessed files -- and the
    # SASRecDataset/RelationAwareSASRecDataset code paths, which never look at
    # this flag -- are completely unaffected either way.
    args.has_real_timestamps = user_seq_times is not None

    # save model args
    # NOTE: seed and outseq_rel_loss_weight are part of the name so that repeated
    # runs differing only by seed (multi-seed experiments) or by the inter-sequence
    # loss weight get distinct log files and checkpoints instead of silently
    # appending to / overwriting each other.
    args_str = f'{args.model_name}-{args.data_name}-{args.hidden_size}-{args.num_hidden_layers}-{args.num_attention_heads}-{args.hidden_act}-{args.attention_probs_dropout_prob}-{args.hidden_dropout_prob}-{args.max_seq_length}-{args.lr}-{args.weight_decay}-{args.rel_loss_weight}-{args.outseq_rel_loss_weight}-{args.ckp}-seed{args.seed}'
    if args.model_name == 'DynamicRelationAwareSASRecModel':
        # Distinguish D-MT4SR ablation runs (which flags were on, and whether
        # this run's preprocessed data actually carried real timestamps) in
        # the log/checkpoint filename.
        args_str += f'-timedecay{args.use_time_decay}-realtimes{args.has_real_timestamps}-dynloss{args.dynamic_loss_weights}-popneg{args.popularity_neg_sampling}'
        if args.use_time_decay:
            # time_scale / time_decay_floor only affect runs with decay enabled,
            # but when they do, they must appear in the name -- otherwise e.g. a
            # 1-day-scale and a 30-day-scale run overwrite each other.
            args_str += f'-tscale{args.time_scale}-tfloor{args.time_decay_floor}'
            if args.time_decay_log:
                args_str += '-tlog'
        if args.popularity_neg_sampling and args.popneg_mix != 1.0:
            args_str += f'-popmix{args.popneg_mix}'
    args_str = _append_variant_suffixes(args, args_str)
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    # main.py appends to the log, so re-running the same config+seed (e.g. after
    # an interrupted run, or with the runner's --force) would otherwise mix two
    # runs into one file. Rotate any existing log aside first so each execution
    # gets its own clean file and nothing is lost. Skipped for --do_eval, which
    # is meant to append its evaluation to the training run's log.
    if not args.do_eval and os.path.exists(args.log_file):
        rotated = os.path.join(
            args.output_dir,
            args_str + '.prev-' + time.strftime('%Y%m%d-%H%M%S') + '.txt')
        os.rename(args.log_file, rotated)
        print(f'Existing log moved to {rotated}')
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    #args.item2attribute = item2attribute
    # set item score in train set to `0` in validation
    args.train_matrix = valid_rating_matrix

    # save model
    checkpoint = args_str + '.pt'
    args.checkpoint_path = os.path.join(args.output_dir, checkpoint)

    if args.model_name not in ('RelationAwareSASRecModel', 'DynamicRelationAwareSASRecModel'):
        train_dataset = SASRecDataset(args, user_seq, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = SASRecDataset(args, user_seq, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)
        #eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=200)

        test_dataset = SASRecDataset(args, user_seq, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
        #test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=200)
    elif args.model_name == 'RelationAwareSASRecModel':
        train_dataset = RelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item, data_type='train')
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = RelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item, data_type='valid')
        eval_sampler = SequentialSampler(eval_dataset)

        test_dataset = RelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item, data_type='test')
        test_sampler = SequentialSampler(test_dataset)
    else:
        # D-MT4SR: same relation-aware dataset as MT4SR, plus an optional
        # popularity-aware negative sampling distribution and (if this run's
        # preprocessed data includes them) real per-interaction timestamps,
        # shared across splits.
        item_freq = compute_item_popularity(user_seq, args.item_size - 2)
        sampling_probs = build_popularity_sampling_probs(item_freq) if args.popularity_neg_sampling else None

        train_dataset = DynamicRelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item, data_type='train', sampling_probs=sampling_probs, user_seq_times=user_seq_times)
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

        eval_dataset = DynamicRelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item, data_type='valid', sampling_probs=sampling_probs, user_seq_times=user_seq_times)
        eval_sampler = SequentialSampler(eval_dataset)

        test_dataset = DynamicRelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, relationships_ind_map, Item, data_type='test', sampling_probs=sampling_probs, user_seq_times=user_seq_times)
        test_sampler = SequentialSampler(test_dataset)


    if args.model_name == 'DistSAModel':
        model = DistSAModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'DistMeanSAModel':
        model = DistMeanSAModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=100)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=100)
        trainer = DistSAModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'RelationAwareSASRecModel': 
        model = RelationAwareSASRecModel(args, relationships_ind_map)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        trainer = RelationAwareSASRecModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    elif args.model_name == 'DynamicRelationAwareSASRecModel':
        model = DynamicRelationAwareSASRecModel(args, relationships_ind_map)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)
        trainer = DynamicRelationAwareSASRecModelTrainer(model, train_dataloader, eval_dataloader,
                                    test_dataloader, args)
    else:
        model = SASRecModel(args=args)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

        trainer = FinetuneTrainer(model, train_dataloader, eval_dataloader,
                                test_dataloader, args)


    if args.do_eval:
        trainer.load(args.checkpoint_path)
        print(f'Load model from {args.checkpoint_path} for test!')
        #scores, result_info, _ = trainer.test(0, full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.complicated_eval(user_seq, args)

    else:
        #pretrained_path = os.path.join(args.output_dir, f'{args.data_name}-epochs-{args.ckp}.pt')
        #try:
        #    trainer.load(pretrained_path)
        #    print(f'Load Checkpoint From {pretrained_path}!')

        #except FileNotFoundError:
        #    print(f'{pretrained_path} Not Found! The Model is same as SASRec')

        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience,
                                       verbose=True,
                                       smooth_window=args.val_smooth_window)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            # Evaluate on MRR. With --ema_decay set, validation and
            # checkpointing both use the averaged weights: the live weights are
            # swapped out, scored, checkpointed if best, then swapped back in
            # so training continues from the raw (un-averaged) trajectory.
            # Training itself is untouched either way.
            use_ema = getattr(trainer, 'ema', None) is not None
            if use_ema:
                trainer.ema.copy_to(trainer.model)
            try:
                scores, _, _ = trainer.valid(epoch, full_sort=True)
                early_stopping(np.array(scores[-1:]), trainer.model)
            finally:
                if use_ema:
                    trainer.ema.restore(trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print('---------------Change to test_rating_matrix!-------------------')
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        valid_scores, _, _ = trainer.valid('best', full_sort=True)
        trainer.args.train_matrix = test_rating_matrix
        scores, result_info, _ = trainer.test('best', full_sort=True)

    print(args_str)
    #print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')
main()
