# Contribution log - Nardeen Samy (202301506)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: TrOCR (fine-tuned).** I owned the pretrained TrOCR model: fine-tuning microsoft/trocr-base-stage1 on IAM with mixed precision and gradient checkpointing, checkpoint selection on validation CER, and test evaluation with saved predictions. This model tests hypothesis H4.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log

| Date | Hours | Task | Evidence |
|---|---:|---|---|
|16/11|4 hours| Set up TrOCR fine-tuning (AdamW, lr 1e-5 → 5e-6, FP16, gradient accumulation 4) | |
|16/11|1 hour| Fine-tuned on a Tesla T4 (≈67.5 min) and kept the best validation epoch (3) | |
|16/11|3.5 hours| Evaluated on all 2,915 test lines and saved `trocr_test_predictions.json` | |
|16/11|3 hours| Uploaded `trocr_best.pt` (1.43 GB) to Google Drive | |
|17/11|1 hour| Shared: Data pipeline and `docs/DATASET.md` | |
|17/11|5 hours| Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
|17/11|2 hours| Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
|18/11|1 hour| Shared: Streamlit demo (`demo/app.py`) | |
|18/11|30 mins| Shared: Technical report | |
|18/11|30 mins| Shared: Presentation | |
|19/11|30 mins| Shared: README and repository organisation | |
| | 21 hour  | | |

## 3. Code, experiments and documents I produced

- Code: `trocr_train.py`, `checkpoints/trocr_processor/`
- Experiments: `trocr_best.pt`, `trocr_history.json`, `trocr_test_predictions.json`
- Compute recorded in the logs: ≈67.5 min of recorded GPU fine-tuning time (Tesla T4)
- Report: §13 (TrOCR results) and TrOCR parts of §6 and §10
- Presentation: Slides 9 and 14 (TrOCR fine-tuning, conclusion) and the live demo

## 4. Hardest problem I solved

Fitting a 385-million-parameter model on a single 16 GB Tesla T4. I used batch size 2 with 4-step gradient accumulation (effective batch 8), FP16 mixed precision and gradient checkpointing. Validation CER got worse in epoch 2 (5.77% → 6.53%), so the learning rate was lowered from 1e-5 to 5e-6, and epoch 3 gave the best result (4.86%). The final checkpoint was 1.43 GB, too large for GitHub, so we host it on Google Drive.

## 5. What I learned

How transfer learning changes the picture: starting from pretrained weights, TrOCR reached 7.0% test CER versus 19.4% for our best model trained from scratch. I learned practical memory-saving techniques for large models and why checkpoints must be selected on validation, not test, data.

