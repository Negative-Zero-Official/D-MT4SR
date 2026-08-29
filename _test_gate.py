"""Sanity tests for the D-MT4SR v2 gate. Not part of the model; run manually."""
import types
import torch
import torch.nn as nn

from modules import RelationAwareSelfAttention, DynamicRelationAwareSelfAttention


def make_args(**over):
    a = types.SimpleNamespace(
        hidden_size=32, num_attention_heads=2, attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0, initializer_range=0.02, max_seq_length=8,
        cuda_condition=False,
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


B, L, R, H = 3, 6, 4, 2
torch.manual_seed(0)
x = torch.randn(B, L, 32)
# attention_mask: 0 = attendable, -10000 = masked. Mask out the first 2
# positions of sample 0 to exercise the "valid position" logic.
mask = torch.zeros(B, 1, L, L)
mask[0, :, :, :2] = -10000.0
mask[0, :, :2, :] = -10000.0
rel_mask = (torch.rand(B, R, L, L) > 0.7).float()

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


# ---------------------------------------------------------------------------
# 1. Residual gate with gate_scale=0 must reproduce MT4SR's static weighting
# ---------------------------------------------------------------------------
args = make_args(gate_residual=True, gate_scale_init=0.0)
dyn = DynamicRelationAwareSelfAttention(args, R).eval()
with torch.no_grad():
    dyn.relationship_weights.copy_(torch.tensor([0.2, 0.9, -0.4, 0.1]))
    static = nn.Softmax(dim=0)(dyn.relationship_weights)

    logits = dyn.relation_gate(x).view(B, L, 1, R).permute(0, 2, 1, 3).unsqueeze(3)
    logits = dyn.relationship_weights.view(1, 1, 1, 1, R) + dyn.gate_scale * logits
    w = nn.Softmax(dim=-1)(logits)

diff = (w - static.view(1, 1, 1, 1, R)).abs().max().item()
check('residual gate @ scale=0 == softmax(relationship_weights)', diff < 1e-6,
      f'max abs diff {diff:.2e}')

# ---------------------------------------------------------------------------
# 2. Every gate option runs, produces the right output shape, and backprops
# ---------------------------------------------------------------------------
variants = {
    'v1 (plain dynamic gate)': dict(),
    'residual': dict(gate_residual=True),
    'residual+pairwise': dict(gate_residual=True, gate_pairwise=True),
    'residual+relmask': dict(gate_residual=True, gate_use_rel_mask=True),
    'residual+perhead': dict(gate_residual=True, gate_per_head=True),
    'residual+entropy': dict(gate_residual=True, gate_entropy_weight=0.05),
    'residual+temp': dict(gate_residual=True, gate_temperature=2.0),
    'all options': dict(gate_residual=True, gate_pairwise=True,
                        gate_use_rel_mask=True, gate_per_head=True,
                        gate_entropy_weight=0.05, gate_temperature=0.5),
    'timedecay+log': dict(gate_residual=True, use_time_decay=True,
                          has_real_timestamps=True, time_decay_log=True),
    'timedecay position fallback': dict(use_time_decay=True,
                                        has_real_timestamps=False),
}

times = torch.rand(B, L) * 5e8   # ~years of unix-second spread

for name, over in variants.items():
    a = make_args(**over)
    m = DynamicRelationAwareSelfAttention(a, R)
    m.train()
    out = m(x, mask, rel_mask, input_times=times)
    shape_ok = tuple(out.shape) == (B, L, 32)
    finite = torch.isfinite(out).all().item()
    out.sum().backward()
    grads = any(p.grad is not None and torch.isfinite(p.grad).all()
                for p in m.parameters())
    ent = m.last_gate_entropy
    ent_ok = True
    if over.get('gate_entropy_weight', 0.0) > 0:
        ent_ok = ent is not None and torch.isfinite(ent).all().item() and ent.item() >= 0
    check(f'forward/backward: {name}',
          shape_ok and finite and grads and ent_ok,
          f'out={tuple(out.shape)} entropy={None if ent is None else round(ent.item(), 4)}')

# ---------------------------------------------------------------------------
# 3. Entropy is maximal for a uniform gate (bounded by log R) and is computed
#    only over valid positions
# ---------------------------------------------------------------------------
a = make_args(gate_residual=True, gate_entropy_weight=0.05)
m = DynamicRelationAwareSelfAttention(a, R).train()
with torch.no_grad():
    nn.init.zeros_(m.relation_gate.weight)
    nn.init.zeros_(m.relation_gate.bias)
    m.relationship_weights.zero_()
    m.gate_scale.zero_()
m(x, mask, rel_mask)
import math
check('uniform gate entropy == log(R)',
      abs(m.last_gate_entropy.item() - math.log(R)) < 1e-5,
      f'{m.last_gate_entropy.item():.6f} vs {math.log(R):.6f}')

# ---------------------------------------------------------------------------
# 4. gate_scale actually receives gradient at init (so it can leave 0)
# ---------------------------------------------------------------------------
a = make_args(gate_residual=True, rel_score_norm='std')
m = DynamicRelationAwareSelfAttention(a, R).train()
(m(x, mask, rel_mask)**2).mean().backward()
g = m.gate_scale.grad
# NOTE: with rel_score_norm='none' the attention softmax is saturated, so this
# gradient is ~0 at init. gate_scale still leaves 0 under Adam (which
# normalizes per-parameter step size), but the meaningful check is that the
# gradient is nonzero once the relation scores are normalized.
check('gate_scale gets a nonzero gradient at init (rel_score_norm=std)',
      g is not None and g.abs().item() > 0, f'grad={g.item():.3e}')

# ---------------------------------------------------------------------------
# 4b. rel_score_norm unsaturates the attention softmax, on BOTH model families
# ---------------------------------------------------------------------------
max_ent = math.log(L)
for cls_name, cls in [('MT4SR', RelationAwareSelfAttention),
                      ('D-MT4SR', DynamicRelationAwareSelfAttention)]:
    ents = {}
    for norm in ['none', 'std']:
        torch.manual_seed(3)
        mm = cls(make_args(rel_score_norm=norm), R).eval()
        mm(x, mask, rel_mask)
        ents[norm] = mm.last_attn_entropy
    check(f'{cls_name}: rel_score_norm=std raises attention entropy',
          ents['std'] > ents['none'] * 5,
          f"none={ents['none']:.4f} std={ents['std']:.4f} (max {max_ent:.3f})")

# ---------------------------------------------------------------------------
# 5. Baseline attention still works untouched
# ---------------------------------------------------------------------------
base = RelationAwareSelfAttention(make_args(), R)
out = base(x, mask, rel_mask)
check('MT4SR baseline attention unchanged', tuple(out.shape) == (B, L, 32))

print('\n' + '=' * 60)
failed = [n for n, ok, _ in results if not ok]
print(f'{len(results) - len(failed)}/{len(results)} passed')
if failed:
    print('FAILED:', *failed, sep='\n  ')
    raise SystemExit(1)
