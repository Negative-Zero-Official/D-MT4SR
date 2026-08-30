"""Non-personalized popularity baseline, evaluated exactly like the model.

Ranks every item by its frequency in the TRAINING portion of the sequences
(everything except each user's last two interactions), masks out each user's
own training items, and scores the held-out test target with the same
HIT@k / NDCG@k / MRR definitions main.py uses.

This is the floor any recommender must clear. It uses no sequence information,
no relations, and no learning at all -- it recommends the same list to every
user. If a sequential relation-aware model does not beat it by a clear margin,
the reported numbers are measuring dataset skew rather than model quality.

Usage:
    python popularity_baseline.py --data_name All_Beauty
    python popularity_baseline.py --data_name Appliances
"""
import argparse
from collections import Counter

import numpy as np

from utils import get_user_seqs_MoHRdata

KS = (1, 5, 10, 15, 20, 40)


def evaluate(user_seq, num_items, split='test'):
    """split='test' uses seq[-1] as target and seq[:-1] as seen;
       split='valid' uses seq[-2] and seq[:-2]."""
    # Train popularity: everything except the last two interactions of every
    # user, so no validation or test information leaks into the ranking.
    pop = Counter()
    for s in user_seq:
        if len(s) >= 3:
            pop.update(s[:-2])

    pop_scores = np.zeros(num_items + 2, dtype=np.float64)
    for item, count in pop.items():
        if 0 <= item < len(pop_scores):
            pop_scores[item] = count
    pop_scores[0] = -np.inf          # padding index is not a real item

    hits = {k: 0 for k in KS}
    ndcg = {k: 0.0 for k in KS}
    rr_sum = 0.0
    n = 0

    for s in user_seq:
        if len(s) < 3:
            continue
        if split == 'test':
            target, seen = s[-1], s[:-1]
        else:
            target, seen = s[-2], s[:-2]

        scores = pop_scores.copy()
        scores[list(seen)] = -np.inf   # same masking the model's eval applies

        # Rank of the target among all unseen items (1-indexed).
        rank = int((scores > scores[target]).sum()) + 1

        rr_sum += 1.0 / rank
        for k in KS:
            if rank <= k:
                hits[k] += 1
                ndcg[k] += 1.0 / np.log2(rank + 1)
        n += 1

    return {
        'n': n,
        'MRR': rr_sum / n,
        **{f'HIT@{k}': hits[k] / n for k in KS},
        **{f'NDCG@{k}': ndcg[k] / n for k in KS},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_name', default='All_Beauty')
    cli = ap.parse_args()

    (user_seq, max_item, _v, _t, num_users,
     _masks, rel_map, _Item, _times) = get_user_seqs_MoHRdata(cli.data_name)

    print(f"dataset       : {cli.data_name}")
    print(f"users         : {num_users}   items: {max_item}   "
          f"relationships: {len(rel_map)}")

    for split in ('valid', 'test'):
        r = evaluate(user_seq, max_item, split=split)
        print(f"\n--- POPULARITY BASELINE ({split}, n={r['n']}) ---")
        print(f"  MRR      {r['MRR']:.4f}")
        for k in (1, 5, 10, 20):
            print(f"  HIT@{k:<3} {r[f'HIT@{k}']:.4f}    NDCG@{k:<3} {r[f'NDCG@{k}']:.4f}")

    print("\n" + "=" * 66)
    print("Compare the TEST row against your model's reported test numbers.")
    print("A sequential, relation-aware model that does not clearly beat a")
    print("non-personalized popularity list is not demonstrating what the")
    print("paper claims, regardless of how the ablations compare to each other.")


if __name__ == '__main__':
    main()
