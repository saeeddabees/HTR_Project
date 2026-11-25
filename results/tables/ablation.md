# Ablation study (same protocol: 10 epochs, batch 4, lr 1e-3, seeds 0-2; validation split)

| Configuration                     | Removed / added   | Val CER %    | Val WER %    |   Δ CER vs full (pp) |
|:----------------------------------|:------------------|:-------------|:-------------|---------------------:|
| BiLSTM, attention removed         | - attention       | 18.17 ± 0.41 | 53.04 ± 1.08 |                 1.45 |
| BiLSTM + attention (reference)    | -                 | 16.72 ± 0.25 | 50.29 ± 0.31 |                 0    |
| BiLSTM + attention + augmentation | + augmentation    | 14.26 ± 0.71 | 45.45 ± 1.44 |                -2.45 |
