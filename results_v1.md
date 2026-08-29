# Results

After running [run_experiments.py](run_experiments.py), the results printed to console are as follows:

```plaintext
Planned runs: 24  (already completed: 0, to run now: 24)
Datasets: All_Beauty, Appliances
Main configs (all seeds [1, 7, 42]): baseline, v1, dynloss
Supporting configs (seed 42 only): popneg, timedecay, dynloss_popneg
Output dir: paper_runs/

----------------------------------------------------------------------
[1/24] All_Beauty__baseline__seed1
  Original MT4SR
  console -> paper_runs/console\All_Beauty__baseline__seed1.log
  done in 11m48s
----------------------------------------------------------------------
[2/24] All_Beauty__baseline__seed7
  Original MT4SR
  console -> paper_runs/console\All_Beauty__baseline__seed7.log
  done in 14m17s
----------------------------------------------------------------------
[3/24] All_Beauty__baseline__seed42
  Original MT4SR
  console -> paper_runs/console\All_Beauty__baseline__seed42.log
  done in 10m48s
----------------------------------------------------------------------
[4/24] All_Beauty__v1__seed1
  D-MT4SR: dynamic relation weighting only
  console -> paper_runs/console\All_Beauty__v1__seed1.log
  done in 21m34s
----------------------------------------------------------------------
[5/24] All_Beauty__v1__seed7
  D-MT4SR: dynamic relation weighting only
  console -> paper_runs/console\All_Beauty__v1__seed7.log
  done in 9m58s
----------------------------------------------------------------------
[6/24] All_Beauty__v1__seed42
  D-MT4SR: dynamic relation weighting only
  console -> paper_runs/console\All_Beauty__v1__seed42.log
  done in 12m43s
----------------------------------------------------------------------
[7/24] All_Beauty__dynloss__seed1
  D-MT4SR + adaptive intra/inter loss weights
  console -> paper_runs/console\All_Beauty__dynloss__seed1.log
  done in 9m13s
----------------------------------------------------------------------
[8/24] All_Beauty__dynloss__seed7
  D-MT4SR + adaptive intra/inter loss weights
  console -> paper_runs/console\All_Beauty__dynloss__seed7.log
  done in 9m04s
----------------------------------------------------------------------
[9/24] All_Beauty__dynloss__seed42
  D-MT4SR + adaptive intra/inter loss weights
  console -> paper_runs/console\All_Beauty__dynloss__seed42.log
  done in 17m19s
----------------------------------------------------------------------
[10/24] All_Beauty__popneg__seed42
  D-MT4SR + popularity-aware negative sampling
  console -> paper_runs/console\All_Beauty__popneg__seed42.log
  done in 4m25s
----------------------------------------------------------------------
[11/24] All_Beauty__timedecay__seed42
  D-MT4SR + real-timestamp relation decay
  console -> paper_runs/console\All_Beauty__timedecay__seed42.log
  done in 12m47s
----------------------------------------------------------------------
[12/24] All_Beauty__dynloss_popneg__seed42
  D-MT4SR + adaptive loss weights + popularity sampling
  console -> paper_runs/console\All_Beauty__dynloss_popneg__seed42.log
  done in 3m37s
----------------------------------------------------------------------
[13/24] Appliances__baseline__seed1
  Original MT4SR
  console -> paper_runs/console\Appliances__baseline__seed1.log
  done in 17m46s
----------------------------------------------------------------------
[14/24] Appliances__baseline__seed7
  Original MT4SR
  console -> paper_runs/console\Appliances__baseline__seed7.log
  done in 15m27s
----------------------------------------------------------------------
[15/24] Appliances__baseline__seed42
  Original MT4SR
  console -> paper_runs/console\Appliances__baseline__seed42.log
  done in 16m45s
----------------------------------------------------------------------
[16/24] Appliances__v1__seed1
  D-MT4SR: dynamic relation weighting only
  console -> paper_runs/console\Appliances__v1__seed1.log
  done in 22m41s
----------------------------------------------------------------------
[17/24] Appliances__v1__seed7
  D-MT4SR: dynamic relation weighting only
  console -> paper_runs/console\Appliances__v1__seed7.log
  done in 13m25s
----------------------------------------------------------------------
[18/24] Appliances__v1__seed42
  D-MT4SR: dynamic relation weighting only
  console -> paper_runs/console\Appliances__v1__seed42.log
  done in 22m49s
----------------------------------------------------------------------
[19/24] Appliances__dynloss__seed1
  D-MT4SR + adaptive intra/inter loss weights
  console -> paper_runs/console\Appliances__dynloss__seed1.log
  done in 26m26s
----------------------------------------------------------------------
[20/24] Appliances__dynloss__seed7
  D-MT4SR + adaptive intra/inter loss weights
  console -> paper_runs/console\Appliances__dynloss__seed7.log
  done in 14m26s
----------------------------------------------------------------------
[21/24] Appliances__dynloss__seed42
  D-MT4SR + adaptive intra/inter loss weights
  console -> paper_runs/console\Appliances__dynloss__seed42.log
  done in 14m17s
----------------------------------------------------------------------
[22/24] Appliances__popneg__seed42
  D-MT4SR + popularity-aware negative sampling
  console -> paper_runs/console\Appliances__popneg__seed42.log
  done in 15m30s
----------------------------------------------------------------------
[23/24] Appliances__timedecay__seed42
  D-MT4SR + real-timestamp relation decay
  console -> paper_runs/console\Appliances__timedecay__seed42.log
  done in 13m48s
----------------------------------------------------------------------
[24/24] Appliances__dynloss_popneg__seed42
  D-MT4SR + adaptive loss weights + popularity sampling
  console -> paper_runs/console\Appliances__dynloss_popneg__seed42.log
  done in 7m22s

======================================================================
Suite finished in 5h38m
Completed runs on record: 24/24

======================================================================
AGGREGATED RESULTS: MRR
======================================================================

Metric: MRR   (sorted best-first)

config                                                                                                                                                                   n      mean      std  runs
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayTrue-realtimesTrue-dynlossFalse-popnegFalse-tscale86400.0-tfloor0.1   1    0.2906   0.0000  0.2906
RelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10                                                                                       3    0.2906   0.0030  0.2929 0.2917 0.2872
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegFalse                           3    0.2897   0.0232  0.3124 0.2904 0.2661
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegFalse                          3    0.2862   0.0129  0.2992 0.2859 0.2734
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegTrue                           1    0.2778   0.0000  0.2778
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegTrue                            1    0.2706   0.0000  0.2706
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegFalse                          3    0.0496   0.0039  0.0534 0.0498 0.0456
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayTrue-realtimesTrue-dynlossFalse-popnegFalse-tscale86400.0-tfloor0.1   1    0.0486   0.0000  0.0486
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegFalse                           3    0.0474   0.0025  0.0495 0.0481 0.0446
RelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10                                                                                       3    0.0459   0.0019  0.0472 0.0468 0.0437
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegTrue                           1    0.0334   0.0000  0.0334
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegTrue                            1    0.0235   0.0000  0.0235

Note: configs with n=1 have no std -- single run, treat differences smaller than the multi-seed spread as inconclusive.

======================================================================
AGGREGATED RESULTS: NDCG@10
======================================================================

Metric: NDCG@10   (sorted best-first)

config                                                                                                                                                                   n      mean      std  runs
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayTrue-realtimesTrue-dynlossFalse-popnegFalse-tscale86400.0-tfloor0.1   1    0.3356   0.0000  0.3356
RelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10                                                                                       3    0.3319   0.0012  0.3333 0.3313 0.3311
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegFalse                           3    0.3298   0.0246  0.3537 0.3310 0.3046
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegFalse                          3    0.3293   0.0163  0.3424 0.3344 0.3111
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegTrue                           1    0.3169   0.0000  0.3169
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegTrue                            1    0.3111   0.0000  0.3111
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegFalse                          3    0.0598   0.0030  0.0629 0.0594 0.0569
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayTrue-realtimesTrue-dynlossFalse-popnegFalse-tscale86400.0-tfloor0.1   1    0.0569   0.0000  0.0569
RelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10                                                                                       3    0.0557   0.0020  0.0574 0.0562 0.0534
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegFalse                           3    0.0553   0.0037  0.0585 0.0561 0.0513
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegTrue                           1    0.0479   0.0000  0.0479
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegTrue                            1    0.0270   0.0000  0.0270

Note: configs with n=1 have no std -- single run, treat differences smaller than the multi-seed spread as inconclusive.

======================================================================
AGGREGATED RESULTS: HIT@10
======================================================================

Metric: HIT@10   (sorted best-first)

config                                                                                                                                                                   n      mean      std  runs
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayTrue-realtimesTrue-dynlossFalse-popnegFalse-tscale86400.0-tfloor0.1   1    0.4936   0.0000  0.4936
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegFalse                          3    0.4840   0.0236  0.5036 0.4907 0.4577
RelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10                                                                                       3    0.4797   0.0104  0.4871 0.4842 0.4678
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegFalse                           3    0.4764   0.0169  0.4921 0.4785 0.4585
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegTrue                           1    0.4563   0.0000  0.4563
DynamicRelationAwareSASRecModel-All_Beauty-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegTrue                            1    0.4484   0.0000  0.4484
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegFalse                          3    0.1054   0.0021  0.1078 0.1046 0.1039
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossFalse-popnegTrue                           1    0.1052   0.0000  0.1052
RelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10                                                                                       3    0.1007   0.0029  0.1039 0.1001 0.0981
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayTrue-realtimesTrue-dynlossFalse-popnegFalse-tscale86400.0-tfloor0.1   1    0.0981   0.0000  0.0981
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegFalse                           3    0.0926   0.0058  0.0981 0.0930 0.0866
DynamicRelationAwareSASRecModel-Appliances-128-1-1-gelu-0.1-0.3-100-0.001-0.0-0.1-0.05-10-timedecayFalse-realtimesTrue-dynlossTrue-popnegTrue                            1    0.0411   0.0000  0.0411

Note: configs with n=1 have no std -- single run, treat differences smaller than the multi-seed spread as inconclusive.
```

dynloss's "win" doesn't replicate
|  | seed 42 (original) | 3-seed mean ± std | individual seeds |
| ----- | ----- | -----| -----|
| dynloss (All_Beauty MRR) | 0.3124 | 0.2897 ± 0.0232 | 0.3124, 0.2904, 0.2661 |
| baseline (All_Beauty MRR) | 0.2929 | 0.2906 ± 0.0030 | 0.2929, 0.2917, 0.2872 |

My seed-42 dynloss run (0.3124) — the one that looked like a clear winner earlier — turns out to be the best of a highly variable set, not representative. Another seed lands at 0.2661, well below baseline. The 3-seed mean (0.2897) is now statistically indistinguishable from baseline (0.2906 ± 0.0030) and from v1 (0.2862 ± 0.0129). All three overlap heavily.

This is precisely the scenario I flagged when trying multi-seed runs. Worth reporting in the paper itself as a methodological point, not hiding it: "a single-seed pilot suggested a large gain from adaptive loss weighting; multi-seed evaluation showed this was not distinguishable from baseline variance."

Where there IS a real, consistent signal: v1 on Appliances
| config (Appliances, n=3) | MRR | NDCG@10 | HIT@10 |
| ----- | ----- | ----- | ----- |
| v1 (dynamic weighting) | 0.0496 ± 0.0039 | 0.0598 ± 0.0030 | 0.1054 ± 0.0021 |
| dynloss | 0.0474 ± 0.0025 | 0.0553 ± 0.0037 | 0.0926 ± 0.0058 |
| baseline | 0.0459 ± 0.0019 | 0.0557 ± 0.0020 | 0.1007 ± 0.0019 |

v1 wins on all three metrics, consistently, across the multi-seed group. It's not a huge effect (v1's worst seed, 0.0456 MRR, dips just below baseline's best, 0.0472), but it's the only config that beats baseline on every metric on this dataset, and it does so while dynloss — last week's apparent star — actually underperforms baseline on HIT@10 here (0.0926 vs 0.1007).