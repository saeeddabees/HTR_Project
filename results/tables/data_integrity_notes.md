# Data-integrity notes

Found automatically while building the results tables.

- `baseline_seed1`: file name says seed 1, but the logged `--seed` argument is 2.
- `attention_aug_seed0/1` logs do not store the `augment` flag (older version of `train.py`); augmentation is inferred from the run name. `attention_aug_seed2` stores `augment: true`.
- The CNN-GRU/BiLSTM main runs used 20 epochs, batch 8; the attention runs used 10 epochs, batch 4. The matched-protocol comparison is the ablation table.
