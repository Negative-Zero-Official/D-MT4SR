"""
Measure whether MT4SR's attention softmax is saturated by the relation scores.

This is the fast go/no-go check. It trains for a handful of epochs and reports,
per epoch:

    rel_score_absmax : magnitude of the relation term added to the attention
                       logits
    attn_score_absmax: magnitude of the ordinary query/key attention logits
    attn_entropy     : entropy of the resulting attention distribution
                       (max = log(max_seq_length), ~4.61 for L=100)

If attn_entropy sits near 0 while rel_score_absmax is orders of magnitude
larger than attn_score_absmax, the attention distribution is effectively
one-hot and driven entirely by the relation pathway -- the ordinary sequential
self-attention is contributing nothing. That is the condition --rel_score_norm
is designed to fix, and this script shows the before/after side by side.

Usage:
    python diagnose_saturation.py --data_name All_Beauty
    python diagnose_saturation.py --data_name Appliances --epochs 3

Runs both rel_score_norm=none and rel_score_norm=std unless --norm is given.
Takes a few minutes, needs no changes to your experiment directory, and writes
nothing except its own console output.
"""
import argparse
import math
import types

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from datasets import RelationAwareSASRecDataset
from seqmodels import RelationAwareSASRecModel
from utils import get_user_seqs_MoHRdata, set_seed


def build_args(cli, rel_score_norm):
    """Mirrors main.py's defaults for the shared MT4SR hyperparameters."""
    a = types.SimpleNamespace(
        data_name=cli.data_name,
        model_name='RelationAwareSASRecModel',
        hidden_size=128, num_hidden_layers=1, num_attention_heads=1,
        hidden_act='gelu', attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.3, initializer_range=0.02,
        max_seq_length=100, lr=0.001, batch_size=128, weight_decay=0.0,
        adam_beta1=0.9, adam_beta2=0.999,
        rel_loss_weight=0.1, outseq_rel_loss_weight=0.05,
        rel_loss_chunk_size=0, no_cuda=cli.no_cuda,
        rel_score_norm=rel_score_norm,
        ema_decay=0.0, gate_lr_scale=1.0, log_freq=1,
    )
    a.cuda_condition = torch.cuda.is_available() and not a.no_cuda
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_name', default='All_Beauty')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-batches', type=int, default=40,
                    help='batches per epoch to time-box the check (0 = all)')
    ap.add_argument('--norm', nargs='+', default=['none', 'std'],
                    choices=['none', 'std', 'layernorm'])
    ap.add_argument('--no_cuda', action='store_true')
    cli = ap.parse_args()

    print(f'Loading {cli.data_name} ...')
    (user_seq, max_item, _valid_m, _test_m, num_users,
     user_seq_mask_mat_rel, relationships_ind_map, Item,
     _times) = get_user_seqs_MoHRdata(cli.data_name)
    print(f'  users={num_users}  items={max_item}  '
          f'relationships={len(relationships_ind_map)}')

    max_ent = math.log(100)
    summary = {}

    for norm in cli.norm:
        print('\n' + '=' * 74)
        print(f'rel_score_norm = {norm}')
        print('=' * 74)
        set_seed(cli.seed)

        args = build_args(cli, norm)
        args.item_size = max_item + 2
        args.num_users = num_users
        args.mask_id = max_item + 1

        ds = RelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel,
                                        relationships_ind_map, Item,
                                        data_type='train')
        dl = DataLoader(ds, sampler=RandomSampler(ds), batch_size=args.batch_size)

        model = RelationAwareSASRecModel(args, relationships_ind_map)
        device = torch.device('cuda' if args.cuda_condition else 'cpu')
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                               betas=(args.adam_beta1, args.adam_beta2))

        attn = model.item_encoder.layer[0].attention

        print(f"{'epoch':>5} {'rel_absmax':>12} {'qk_absmax':>11} "
              f"{'attn_entropy':>13} {'% of max':>9} {'relW drift':>11}")
        print('-' * 74)

        w0 = model.item_encoder.layer[0].attention.relationship_weights.detach().clone()

        for epoch in range(cli.epochs):
            model.train()
            qk_absmax = 0.0
            for i, batch in enumerate(dl):
                if cli.max_batches and i >= cli.max_batches:
                    break
                batch = tuple(t.to(device) for t in batch)
                (_, input_ids, target_pos, target_neg, _,
                 rel_seq_masks, item_rel, item_rel_pos) = batch

                seq_out, _, _, _ = model.finetune(
                    input_ids, rel_seq_masks[:, :, :-1, :-1])

                # Plain next-item BPR-style loss; enough to drive training for
                # a diagnostic. The relation regularizers are omitted on
                # purpose -- they don't touch the attention softmax, which is
                # what we're measuring.
                pos_emb = model.item_embeddings(target_pos)
                neg_emb = model.item_embeddings(target_neg)
                pos_logit = (seq_out * pos_emb).sum(-1)
                neg_logit = (seq_out * neg_emb).sum(-1)
                istarget = (target_pos > 0).float()
                loss = -(torch.log(torch.sigmoid(pos_logit - neg_logit) + 1e-24)
                         * istarget).sum() / istarget.sum().clamp_min(1.0)

                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    q = model.item_encoder.layer[0].attention.query(
                        model.add_position_embedding(input_ids))
                    k = model.item_encoder.layer[0].attention.key(
                        model.add_position_embedding(input_ids))
                    s = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(
                        args.hidden_size / args.num_attention_heads)
                    qk_absmax = max(qk_absmax, float(s.abs().max()))

            drift = float((attn.relationship_weights.detach() - w0.to(device)).norm())
            ent = attn.last_attn_entropy
            print(f"{epoch:>5} {attn.last_rel_score_absmax:>12.2f} "
                  f"{qk_absmax:>11.3f} {ent:>13.4f} "
                  f"{100.0 * ent / max_ent:>8.2f}% {drift:>11.5f}")

        summary[norm] = (attn.last_rel_score_absmax, qk_absmax,
                         attn.last_attn_entropy)

    print('\n' + '=' * 74)
    print('VERDICT')
    print('=' * 74)
    print(f'Maximum possible attention entropy for L=100 is {max_ent:.3f} '
          '(uniform attention).')
    for norm, (rel, qk, ent) in summary.items():
        ratio = rel / qk if qk else float('inf')
        state = ('SATURATED - attention is effectively one-hot'
                 if ent < 0.1 * max_ent else
                 'healthy - attention is a real distribution')
        print(f'  {norm:>9}: relation/ordinary score ratio {ratio:>10.1f}x   '
              f'entropy {ent:.4f} ({100 * ent / max_ent:.1f}% of max)  -> {state}')
    print('\nIf "none" is SATURATED and "std" is healthy, your MT4SR baseline has')
    print('been running with its sequential self-attention drowned out by the')
    print('relation pathway, and --rel_score_norm=std is worth a full suite.')


if __name__ == '__main__':
    main()
