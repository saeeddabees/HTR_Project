# Deliverables checklist

Status of the 12 end-of-semester deliverables. ✅ = in the repo and done · ✏️ = in the repo, team must fill in or finish · 🔗 = hosted outside git.

| # | Deliverable | Required | Where | Status |
|---|---|---|---|---|
| 1 | Source code | Git repo, organised and documented | `src/`, `trocr_train.py`, `demo/`, `scripts/` | ✅ code · ✏️ push the updated folder to the GitHub repo |
| 2 | Dataset documentation | Sources, versions, licences, preprocessing, split strategy | `docs/DATASET.md` | ✅ (optional: record the HF dataset commit hash, §2) |
| 3 | Trained models | Final checkpoints, or scripts that reproduce them | `checkpoints/`, `docs/MODELS.md`, `scripts/reproduce_all.*` | ✅ CTC · 🔗 TrOCR (`trocr_best.pt`, Google Drive) |
| 4 | Experiment results | All runs, with tables and graphs | `results/tables/all_runs.md`, `results/figures/01-08` | ✅ |
| 5 | Model comparison | All approaches, same protocol, verdict per hypothesis | `docs/RESULTS.md` §5, `results/tables/model_comparison.md`, `hypothesis_verdicts.md` | ✅ |
| 6 | Error analysis | Failure cases with examples and explanations | `results/error_analysis/trocr/ERROR_ANALYSIS.md` | ✅ TrOCR · ✏️ optional: run `predict_ctc` + `error_analysis` for the best CTC model (needs torch) |
| 7 | Ablation study | Results and interpretation | `results/tables/ablation.md`, `docs/RESULTS.md` §7 | ✅ |
| 8 | Technical report | 15-25 pages, section-4 structure | `report/HTR_AI403_Technical_Report_Formatted.pdf` | ✅ 17 pages, 9 figures added (sections 10-14) · original kept in `report/archive/` |
| 9 | Presentation | 15 min, every member presents | `presentation/` (online deck, exportable to PPTX/PDF) | ✏️ replace `[Member A-E]` with names, rehearse timing |
| 10 | Live demonstration | 5 min, live | `demo/app.py`, `docs/DEMO_SCRIPT.md` | ✏️ download TrOCR checkpoint, record backup video, rehearse |
| 11 | README | Full run instructions | `README.md` | ✅ |
| 12 | Contribution log + peer evaluation | One per student | `contributions/` | ✏️ each student fills `log_<name>.md` and `peer_<name>.md` from the templates |

## Presentation timing (15 min)

| Part | Slides | Presenter | Time |
|---|---|---|---|
| Problem, hypotheses, dataset, preprocessing | 1-4 | Member A | 3:00 |
| CTC architectures, training protocol | 5-6 | Member B | 2:00 |
| Attention ablation, learning-rate study | 7-8 | Member C | 2:30 |
| TrOCR fine-tuning | 9 | Member D | 1:30 |
| Comparison, verdicts, error analysis | 10-12 | Member E | 3:30 |
| Conclusion + Q&A handoff | 14-15 | All | 1:00 |
| *(Live demo, slide 13 - separate 5 min)* | 13 | Member E | 5:00 |

## Known data-integrity points (be ready if asked)

- `baseline_seed1` was trained with `--seed 2` (same seed as `baseline_seed2`, different results because of GPU non-determinism). It does not change H1.
- The main GRU/BiLSTM runs (20 epochs, batch 8) and attention runs (10 epochs, batch 4) use different budgets; the matched comparison is the ablation.
- TrOCR is a single run.
