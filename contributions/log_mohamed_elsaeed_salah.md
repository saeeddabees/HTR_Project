# Contribution log - Mohamed Elsaeed Salah (202303403)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: CNN-BiLSTM + Multi-Head Attention + CTC.** I owned the attention model: the multi-head self-attention block with residual connection and LayerNorm, training-only data augmentation, the learning-rate study, and training both variants on three seeds. This model tests hypotheses H2 and H3.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log


| Date | Hours | Task | Evidence |
|---|---:|---|---|
|16/11|2 hour| Implemented `CRNNBiLSTMAttention` (4-head self-attention, residual + LayerNorm) | |
|16/11|3 hour| Implemented training-only augmentation (rotation, brightness/contrast, noise) | |
|17/11|3 hours| Trained attention and attention + augmentation on seeds 0-2 | |
|17/11|1.5 hour| Ran the learning-rate study (0.0005 / 0.001 / 0.002) | |
|18/11|1 hour| Shared: Data pipeline and `docs/DATASET.md` | |
|18/11|5 hours| Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
|18/11|2 hours| Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
|19/11|1.5 hour| Shared: Streamlit demo (`demo/app.py`) | |
|19/11|1 hour| Shared: Technical report | |
|19/11|1 hour| Shared: Presentation | |
|19/11|1 hour| Shared: README and repository organisation | |
| | 21 hour | | |

## 3. Code, experiments and documents I produced

- Code: `src/models/attention.py`, `apply_augmentation` in `src/datasets/iam.py`, `src/hyperparameter_search.py`
- Experiments: `attention_seed0`, `attention_seed1`, `attention_seed2`, `attention_aug_seed0`, `attention_aug_seed1`, `attention_aug_seed2`, `hp_attention_lr0005`, `hp_attention_lr001`, `hp_attention_lr002`, `hp_attention_lr002_seed1`, `hp_attention_lr002_seed2`
- Compute recorded in the logs: 124 min of recorded GPU training time across 11 runs
- Report: Attention and augmentation parts of §5, §6, §8.1 (learning-rate study), §11 and §12 (ablation)
- Presentation: Slides 7, 8, 12 (ablation, learning-rate study, error analysis) and the live demo

## 4. Hardest problem I solved

Choosing augmentation that helps rather than hurts. Strong rotations or noise make lines that no longer look like IAM handwriting, so I kept it mild (±2° rotation, 0.9-1.1 brightness/contrast, light noise on 30% of images) and made sure it runs only on the training split, never on validation or test. Getting the attention block to train stably also needed a residual connection and LayerNorm around the multi-head attention.

## 5. What I learned

How self-attention lets every position in the line use the whole line, and how residual + LayerNorm keep it stable. Augmentation gave the biggest single gain (−2.45 CER points on validation), which showed me how much a small dataset overfits. The learning-rate study taught me to screen on one seed and confirm the winner on three.
