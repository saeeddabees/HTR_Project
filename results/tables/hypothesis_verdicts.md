# Verdict on each hypothesis

| Hypothesis                                                  | Evidence                                                                  | Verdict   | Caveat                                                                       |
|:------------------------------------------------------------|:--------------------------------------------------------------------------|:----------|:-----------------------------------------------------------------------------|
| H1: BiLSTM beats the unidirectional CNN-GRU baseline        | Test CER 29.00 → 24.77; WER 70.87 → 62.77                                 | Supported | Baseline seed-1 log records seed=2 (see data-integrity note)                 |
| H2: Self-attention on top of BiLSTM improves recognition    | Ablation val CER 18.17 → 16.72 (matched protocol); test CER 24.77 → 22.29 | Supported | Test comparison uses BiLSTM trained 20 ep/batch 8 vs attention 10 ep/batch 4 |
| H3: Training-only augmentation improves generalisation      | Test CER 22.29 → 19.39; WER 58.47 → 53.61                                 | Supported | Gap (~2.9 pp) is larger than seed SD (≤0.9 pp)                               |
| H4: Fine-tuned pretrained TrOCR beats all custom CTC models | Test CER 7.02 vs best CTC 19.39                                           | Supported | Single run; ~260× more parameters; not compute-matched                       |
