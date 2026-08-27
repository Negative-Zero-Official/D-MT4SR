# MT4SR
This is the implementation for the paper:
BigData'22. You may find it on [Arxiv](https://arxiv.org/pdf/2210.13572.pdf)

The code is built on Pytorch.
The dataset preprocess code is in data/ dir. Change the path in the code and also include the meta data file path.

Code to run:
```python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=relationsasrec_v6/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=RelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weigh=0.1 --outseq_rel_loss_weight=0.05```

## D-MT4SR (Dynamic MT4SR)

D-MT4SR extends MT4SR in-place (no new files; `--model_name=DynamicRelationAwareSASRecModel` selects it in `main.py`) with four independently-toggleable improvements over the original `RelationAwareSASRecModel`:

1. **Dynamic relation weighting** (always on for this model): replaces MT4SR's single global, dataset-wide per-relationship weight (`RelationAwareSelfAttention.relationship_weights`) with a context-conditioned gate (`DynamicRelationAwareSelfAttention.relation_gate` in `modules.py`) that predicts relation importance per sequence position instead of once for the whole dataset.
2. **`--use_time_decay`**: adds a learnable per-relationship decay over item-to-item distance. If the loaded dataset was preprocessed with the updated `preprocess_fromscratch.py` (see below), this uses **real elapsed time** between interactions; otherwise it automatically falls back to sequence-position distance as a proxy.
3. **`--dynamic_loss_weights`**: replaces the static, grid-searched `rel_loss_weight`/`outseq_rel_loss_weight` (α/β) with a learned multiplicative correction on top of them (`log_alpha_scale`/`log_beta_scale` in `DynamicRelationAwareSASRecModel`), optimized jointly via backprop.
4. **`--popularity_neg_sampling`**: replaces uniform random negative sampling with popularity-aware (freq^0.75) sampling for harder, more informative negatives.

With all three flags off, `DynamicRelationAwareSASRecModel` + `DynamicRelationAwareSASRecModelTrainer` reproduce MT4SR's training dynamics except for the dynamic relation-weighting architecture change (#1), which is the core contribution and always active for this model — this makes it straightforward to run controlled ablations against the original `RelationAwareSASRecModel` baseline.

### Real timestamps (`--use_time_decay`)

`preprocess_fromscratch.py` now saves three extra fields (`user_train_times`, `user_validation_times`, `user_testing_times`) alongside the original 8, appended at the **end** of the saved list so the format stays backward compatible:

- **Old `.npy` files** (8 elements, from before this change) still load fine — `utils.get_user_seqs_MoHRdata` detects the shorter format via `len(dataset)` and returns `user_seq_times=None`. `--use_time_decay` then automatically uses the position-distance proxy, exactly as before.
- **Regenerated `.npy` files** (11 elements, via the updated `preprocess_fromscratch.py`) carry real per-interaction unix timestamps through to the model. `main.py` sets `args.has_real_timestamps` based on which format was loaded, and `DynamicRelationAwareSelfAttention` picks the real-timestamp decay path automatically when it's `True`.
- This is fully backward compatible with the **original** `RelationAwareSASRecModel`/`SASRecModel` too — they never read the timestamp fields at all, so old and new preprocessed data both work for every model in this repo unchanged.

To get real timestamp decay, regenerate the preprocessed file:
```bash
cd data/  # or wherever preprocess_fromscratch.py + the raw Amazon files live
python preprocess_fromscratch.py
```
This overwrites `<DATASET>Partitioned_5core.npy` with the 11-element format. No other files or commands change — `main.py` picks up real timestamps automatically once that file is regenerated, for any run using `--model_name=DynamicRelationAwareSASRecModel --use_time_decay`.

Example commands, using the same hyperparameters as the original MT4SR run for direct comparison:

```bash
# Baseline MT4SR (unchanged)
python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=relationsasrec_v6/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=RelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weight=0.1 --outseq_rel_loss_weight=0.05

# D-MT4SR: dynamic relation weighting only
python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=dmt4sr_v1/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=DynamicRelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weight=0.1 --outseq_rel_loss_weight=0.05

# D-MT4SR: + time decay + adaptive loss weights + popularity negative sampling
python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=dmt4sr_full/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=DynamicRelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weight=0.1 --outseq_rel_loss_weight=0.05 --use_time_decay --dynamic_loss_weights --popularity_neg_sampling
```

`--time_scale` (default `86400.0`, i.e. seconds/day) controls the unit real timestamp gaps are normalized to before the learnable decay is applied; only relevant when `--use_time_decay` is set and the loaded data has real timestamps.

Each combination of flags produces a distinctly-named log/checkpoint file (see `args_str` in `main.py`), so results can be compared side by side without overwriting each other.

Please cite our paper if you use the code:
```bibtex
@inproceedings{fan2022sequentialmt4sr,
  title={Sequential Recommendation with Auxiliary Item Relationships via Multi-Relational Transformer},
  author={Fan, Ziwei and Liu, Zhiwei and Wang, Chen and Huang, Peijie and Peng, Hao and Philip, S Yu},
  booktitle={2022 IEEE International Conference on Big Data (Big Data)},
  pages={525--534},
  year={2022},
  organization={IEEE}
}
```
