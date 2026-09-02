# -*- coding: utf-8 -*-
"""Verify that --rel_loss_chunk_size changes nothing but memory.

The handoff runs Office_Products with a chunk size chosen from the detected
VRAM, which is only legitimate if chunking is mathematically neutral. This
script checks that claim directly: it computes the inter-sequence relation
loss on the same batch, with the same weights, unchunked and at several chunk
sizes, and compares both the loss VALUE and the GRADIENTS it produces.

    python verify_chunking.py --data_name=All_Beauty

Exits non-zero if any chunk size disagrees with the unchunked result beyond
float32 round-off. Optional -- preflight does not depend on it -- but it is
the evidence behind "chunk size is safe to change and batch size is not".
"""
import argparse
import sys
import types

import torch

from utils import get_user_seqs_MoHRdata, set_seed
from datasets import RelationAwareSASRecDataset
from seqmodels import RelationAwareSASRecModel
from trainers import RelationAwareSASRecModelTrainer
from torch.utils.data import DataLoader, SequentialSampler

TOL = 2e-5

ap = argparse.ArgumentParser()
ap.add_argument('--data_name', default='All_Beauty')
ap.add_argument('--batch_size', type=int, default=64)
ap.add_argument('--chunks', type=int, nargs='+', default=[8192, 4096, 1024, 256])
cli = ap.parse_args()

(user_seq, max_item, valid_rating_matrix, _, num_users, user_seq_mask_mat_rel,
 rel_map, Item, _) = get_user_seqs_MoHRdata(cli.data_name)

args = types.SimpleNamespace(
    hidden_size=128, num_hidden_layers=1, num_attention_heads=1,
    hidden_act='gelu', attention_probs_dropout_prob=0.0, hidden_dropout_prob=0.0,
    initializer_range=0.02, max_seq_length=100, rel_loss_weight=0.1,
    outseq_rel_loss_weight=0.05, rel_score_norm='none', ema_decay=0.0,
    gate_lr_scale=1.0, lr=0.001, weight_decay=0.0, adam_beta1=0.9,
    adam_beta2=0.999, no_cuda=not torch.cuda.is_available(), batch_size=cli.batch_size,
    item_size=max_item + 2, num_users=num_users, mask_id=max_item + 1,
    train_matrix=valid_rating_matrix, log_file='/dev/null',
    rel_loss_chunk_size=0, data_name=cli.data_name, seed=42,
)
args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

set_seed(42)
ds = RelationAwareSASRecDataset(args, user_seq, user_seq_mask_mat_rel, rel_map,
                                Item, data_type='train')
dl = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=cli.batch_size)
model = RelationAwareSASRecModel(args, rel_map)
trainer = RelationAwareSASRecModelTrainer(model, dl, dl, dl, args)

batch = next(iter(dl))
batch = tuple(t.to(trainer.device) for t in batch)
_, input_ids, _, _, _, rel_seq_masks, item_rel, item_rel_pos = batch

# Dropout off and eval-mode disabled deliberately: we want the TRAINING path,
# including the gradient-checkpoint branch, which only runs when the module is
# in training mode and the inputs require grad.
trainer.model.train()


def loss_and_grad(chunk_size):
    trainer.args.rel_loss_chunk_size = chunk_size
    trainer.model.zero_grad(set_to_none=True)
    _, _, rel_embs, _ = trainer.model.finetune(input_ids, rel_seq_masks[:, :, :-1, :-1])
    loss = trainer.relation_outside_seq_loss(item_rel, item_rel_pos, rel_embs[-1])
    loss.backward()
    grad = trainer.model.item_embeddings.weight.grad.detach().clone()
    return float(loss.item()), grad


set_seed(42)
base_loss, base_grad = loss_and_grad(0)
print(f'unchunked            loss = {base_loss:.10f}   '
      f'|grad| = {base_grad.norm().item():.8f}')

failures = []
for chunk in cli.chunks:
    set_seed(42)
    loss, grad = loss_and_grad(chunk)
    dl_abs = abs(loss - base_loss)
    dg_abs = (grad - base_grad).abs().max().item()
    ok = dl_abs <= TOL and dg_abs <= TOL
    print(f'chunk_size={chunk:<6}       loss = {loss:.10f}   '
          f'|grad| = {grad.norm().item():.8f}   '
          f'dloss = {dl_abs:.2e}  max dgrad = {dg_abs:.2e}   '
          f'{"OK" if ok else "MISMATCH"}')
    if not ok:
        failures.append((chunk, dl_abs, dg_abs))

print()
if failures:
    print('FAIL: chunking changed the result. Do NOT rely on --rel_loss_chunk_size.')
    for chunk, dl_abs, dg_abs in failures:
        print(f'  chunk {chunk}: dloss={dl_abs:.2e} max dgrad={dg_abs:.2e} (tol {TOL:.0e})')
    sys.exit(1)
print(f'PASS: every chunk size matched the unchunked loss and gradients '
      f'within {TOL:.0e}.')
print('Chunk size is therefore safe to vary with the hardware; batch size is not.')
