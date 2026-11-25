# Contribution log - Mohamed Elsaeed Salah (202303403)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: CNN-BiLSTM + Multi-Head Attention + CTC.** I owned the attention model: the multi-head self-attention block with residual connection and LayerNorm, training-only data augmentation, the learning-rate study, and training both variants on three seeds. This model tests hypotheses H2 and H3.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log

Fill in the date and the hours you actually spent on each task.

| Date | Hours | Task | Evidence |
|---|---:|---|---|
| | | Implemented `CRNNBiLSTMAttention` (4-head self-attention, residual + LayerNorm) | |
| | | Implemented training-only augmentation (rotation, brightness/contrast, noise) | |
| | | Trained attention and attention + augmentation on seeds 0-2 | |
| | | Ran the learning-rate study (0.0005 / 0.001 / 0.002) | |
| | | Shared: Data pipeline and `docs/DATASET.md` | |
| | | Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
| | | Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
| | | Shared: Streamlit demo (`demo/app.py`) | |
| | | Shared: Technical report | |
| | | Shared: Presentation | |
| | | Shared: README and repository organisation | |
| | **Total** | | |

## 3. Code, experiments and documents I produced

- Code: `src/models/attention.py`, `apply_augmentation` in `src/datasets/iam.py`, `src/hyperparameter_search.py`
- Experiments: `attention_seed0`, `attention_seed1`, `attention_seed2`, `attention_aug_seed0`, `attention_aug_seed1`, `attention_aug_seed2`, `hp_attention_lr0005`, `hp_attention_lr001`, `hp_attention_lr002`, `hp_attention_lr002_seed1`, `hp_attention_lr002_seed2`
- Compute recorded in the logs: 124 min of recorded GPU training time across 11 runs
- Report: Attention and augmentation parts of §5, §6, §8.1 (learning-rate study), §11 and §12 (ablation)
- Presentation: Slides 7, 8, 12 (ablation, learning-rate study, error analysis) and the live demo

## 4. Hardest problem I solved

<!-- What went wrong, how you found it, how you fixed it. -->

## 5. What I learned

<!-- 3-5 sentences. -->

## 6. Declaration

I confirm this log is an accurate record of my own contribution.

Signature: ____________________  Date: ____________
