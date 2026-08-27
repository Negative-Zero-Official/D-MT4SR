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
2. **`--use_time_decay`**: adds a learnable per-relationship decay over sequence-position distance, modulating how much a relationship contributes based on how far apart two items are in the sequence.
3. **`--dynamic_loss_weights`**: replaces the static, grid-searched `rel_loss_weight`/`outseq_rel_loss_weight` (α/β) with a learned multiplicative correction on top of them (`log_alpha_scale`/`log_beta_scale` in `DynamicRelationAwareSASRecModel`), optimized jointly via backprop.
4. **`--popularity_neg_sampling`**: replaces uniform random negative sampling with popularity-aware (freq^0.75) sampling for harder, more informative negatives.

With all three flags off, `DynamicRelationAwareSASRecModel` + `DynamicRelationAwareSASRecModelTrainer` reproduce MT4SR's training dynamics except for the dynamic relation-weighting architecture change (#1), which is the core contribution and always active for this model — this makes it straightforward to run controlled ablations against the original `RelationAwareSASRecModel` baseline.

Example commands, using the same hyperparameters as the original MT4SR run for direct comparison:

```bash
# Baseline MT4SR (unchanged)
python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=relationsasrec_v6/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=RelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weight=0.1 --outseq_rel_loss_weight=0.05

# D-MT4SR: dynamic relation weighting only
python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=dmt4sr_v1/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=DynamicRelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weight=0.1 --outseq_rel_loss_weight=0.05

# D-MT4SR: + time decay + adaptive loss weights + popularity negative sampling
python main.py --data_name=Beauty --lr=0.001 --hidden_size=128 --output_dir=dmt4sr_full/ --max_seq_length=100 --hidden_dropout_prob=0.3 --num_hidden_layers=1 --weight_decay=0.0 --num_attention_heads=1 --model_name=DynamicRelationAwareSASRecModel --attention_probs_dropout_prob=0.1 --rel_loss_weight=0.1 --outseq_rel_loss_weight=0.05 --use_time_decay --dynamic_loss_weights --popularity_neg_sampling
```

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
