"""How concentrated are the held-out test targets?

If a small number of items account for a large share of test targets, then
HIT@k moves in blocks: every user holding that item flips the moment its score
crosses the rank-k threshold. That produces the staircase pattern seen in the
epoch logs (HIT@40 jumping 0.45 -> 0.73 in one epoch, then HIT@20, then HIT@15)
without anything being wrong in the code.

Usage:  python check_target_concentration.py --data_name All_Beauty
"""
import argparse
from collections import Counter

from utils import get_user_seqs_MoHRdata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_name', default='All_Beauty')
    args = ap.parse_args()

    (user_seq, max_item, _v, _t, num_users,
     _masks, rel_map, _Item, _times) = get_user_seqs_MoHRdata(args.data_name)

    # Leave-one-out: last item is test, second-to-last is validation.
    test_targets = [s[-1] for s in user_seq if len(s) >= 3]
    val_targets = [s[-2] for s in user_seq if len(s) >= 3]

    print(f"dataset            : {args.data_name}")
    print(f"users              : {num_users}")
    print(f"items              : {max_item}")
    print(f"relationships      : {len(rel_map)}")
    print(f"users w/ test tgt  : {len(test_targets)}")

    for label, targets in (('TEST', test_targets), ('VALID', val_targets)):
        c = Counter(targets)
        n = len(targets)
        print(f"\n--- {label} targets ---")
        print(f"  distinct target items : {len(c)}")
        print(f"  top 1  item covers    : {c.most_common(1)[0][1] / n:6.1%}")
        print(f"  top 5  items cover    : {sum(v for _, v in c.most_common(5)) / n:6.1%}")
        print(f"  top 10 items cover    : {sum(v for _, v in c.most_common(10)) / n:6.1%}")
        print(f"  top 20 items cover    : {sum(v for _, v in c.most_common(20)) / n:6.1%}")
        print("  most common (item, users):")
        for item, cnt in c.most_common(8):
            print(f"    item {item:>6}  {cnt:>5} users  ({cnt / n:5.1%})")

    c = Counter(test_targets)
    top_share = sum(v for _, v in c.most_common(10)) / len(test_targets)
    print("\n" + "=" * 62)
    if top_share > 0.20:
        print(f"CONCENTRATED: 10 items account for {top_share:.1%} of test targets.")
        print("HIT@k will move in large discrete blocks as those few items cross")
        print("the rank-k threshold. Per-epoch HIT@k jumps are expected, and")
        print("HIT@k differences between configs are dominated by a handful of")
        print("items rather than by general ranking quality.")
        print("\nReport MRR and NDCG as primary. Treat HIT@k as unstable on this")
        print("dataset, and say so explicitly in the paper.")
    else:
        print(f"NOT concentrated: 10 items account for only {top_share:.1%}.")
        print("Block behaviour is NOT explained by target concentration --")
        print("something else is producing the staircase, keep digging.")


if __name__ == '__main__':
    main()
