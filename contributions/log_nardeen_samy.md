# Contribution log - Nardeen Samy (202301506)

Team: Handwritten Text Recognition on IAM (AI403, Fall 2026)

## 1. My responsibilities

**Model owned: TrOCR (fine-tuned).** I owned the pretrained TrOCR model: fine-tuning microsoft/trocr-base-stage1 on IAM with mixed precision and gradient checkpointing, checkpoint selection on validation CER, and test evaluation with saved predictions. This model tests hypothesis H4.

Shared equally with the team: Data pipeline and `docs/DATASET.md`; Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding); Results, graphs and error analysis (`src/analysis/`, `results/`); Streamlit demo (`demo/app.py`); Technical report; Presentation; README and repository organisation.

## 2. Work log

Fill in the date and the hours you actually spent on each task.

| Date | Hours | Task | Evidence |
|---|---:|---|---|
| | | Set up TrOCR fine-tuning (AdamW, lr 1e-5 → 5e-6, FP16, gradient accumulation 4) | |
| | | Fine-tuned on a Tesla T4 (≈67.5 min) and kept the best validation epoch (3) | |
| | | Evaluated on all 2,915 test lines and saved `trocr_test_predictions.json` | |
| | | Uploaded `trocr_best.pt` (1.43 GB) to Google Drive | |
| | | Shared: Data pipeline and `docs/DATASET.md` | |
| | | Shared: Training / evaluation framework (`train.py`, `test.py`, metrics, CTC decoding) | |
| | | Shared: Results, graphs and error analysis (`src/analysis/`, `results/`) | |
| | | Shared: Streamlit demo (`demo/app.py`) | |
| | | Shared: Technical report | |
| | | Shared: Presentation | |
| | | Shared: README and repository organisation | |
| | **Total** | | |

## 3. Code, experiments and documents I produced

- Code: `trocr_train.py`, `checkpoints/trocr_processor/`
- Experiments: `trocr_best.pt`, `trocr_history.json`, `trocr_test_predictions.json`
- Compute recorded in the logs: ≈67.5 min of recorded GPU fine-tuning time (Tesla T4)
- Report: §13 (TrOCR results) and TrOCR parts of §6 and §10
- Presentation: Slides 9 and 14 (TrOCR fine-tuning, conclusion) and the live demo

## 4. Hardest problem I solved

<!-- What went wrong, how you found it, how you fixed it. -->

## 5. What I learned

<!-- 3-5 sentences. -->

## 6. Declaration

I confirm this log is an accurate record of my own contribution.

Signature: ____________________  Date: ____________
