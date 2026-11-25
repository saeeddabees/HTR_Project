# Contribution log - Mahmoud Mohamed (202303382)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: CNN-BiLSTM + CTC.** I owned the CNN-BiLSTM + CTC model: architecture, training on three seeds, test evaluation, and the BiLSTM runs used as the no-attention ablation. This model tests hypothesis H1.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log


| Date | Hours | Task | Evidence |
|---|---:|---|---|
|16/11|50 min| Implemented `CRNNBiLSTM` | |
|16/11|4 hours| Trained the main BiLSTM (20 epochs, batch 8) on seeds 0-2 | |
|16/11|3.15 hours| Trained the ablation BiLSTM (10 epochs, batch 4) on seeds 0-2 to match the attention protocol | |
|17/11|2.45 hours| Evaluated the BiLSTM checkpoints on the test split | |
|17/11|1 hours | Shared: Data pipeline and `docs/DATASET.md` | |
|17/11|4.75 hours| Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
|18/11|2.30 hours| Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
|18/11|1.5 hours| Shared: Streamlit demo (`demo/app.py`) | |
|19/11|2 hours| Shared: Technical report | |
|19/11|1 hours| Shared: Presentation | |
|19/11|1 hours| Shared: README and repository organisation | |
| | 23.65 hours | | |

## 3. Code, experiments and documents I produced

- Code: `src/models/crnn.py`
- Experiments: `bilstm_seed0`, `bilstm_seed1`, `bilstm_seed2`, `ablation_bilstm_seed0`, `ablation_bilstm_seed1`, `ablation_bilstm_seed2`
- Compute recorded in the logs: 81 min of recorded GPU training time across 6 runs
- Report: BiLSTM parts of §6, §10, §11 and the ablation baseline in §12
- Presentation: Slides 6, 10, 11: training protocol, model comparison, hypothesis verdicts

## 4. Hardest problem I solved

Making the BiLSTM comparison fair. The main BiLSTM runs used 20 epochs and batch 8, but the attention runs used 10 epochs and batch 4, so comparing their test scores mixed up the effect of attention with the effect of the training budget. I trained three extra BiLSTM runs (`ablation_bilstm_seed*`) with exactly the attention protocol, which gave a clean ablation: removing attention costs 1.45 CER points on validation.

## 5. What I learned


Why bidirectional context helps handwriting: the BiLSTM cut test CER from 29.0% to 24.8% over the GRU baseline. I also learned to control an experiment - change one thing at a time - and that a fair comparison sometimes needs extra runs rather than reusing old ones.


