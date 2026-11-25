# Deliverables checklist

Status of the 12 end-of-semester deliverables. ✅ = in the repo and done · ✏️ = in the repo, team must fill in or finish · 🔗 = hosted outside git.

| # | Deliverable | Required | Where | Status |
|---|---|---|---|---|
| 1 | Source code | Git repo, organised and documented | `src/`, `trocr_train.py`, `demo/`, `scripts/` | ✅ on GitHub (`saeeddabees/HTR_Project`, branch `main`) |
| 2 | Dataset documentation | Sources, versions, licences, preprocessing, split strategy | `docs/DATASET.md` | ✅ (optional: record the HF dataset commit hash, §2) |
| 3 | Trained models | Final checkpoints, or scripts that reproduce them | `checkpoints/`, `docs/MODELS.md`, `scripts/reproduce_all.*` | ✅ CTC · 🔗 TrOCR (`trocr_best.pt`, Google Drive) |
| 4 | Experiment results | All runs, with tables and graphs | `results/tables/all_runs.md`, `results/figures/01-08` | ✅ |
| 5 | Model comparison | All approaches, same protocol, verdict per hypothesis | `docs/RESULTS.md` §5, `results/tables/model_comparison.md`, `hypothesis_verdicts.md` | ✅ |
| 6 | Error analysis | Failure cases with examples and explanations | `results/error_analysis/trocr/ERROR_ANALYSIS.md` | ✅ TrOCR · ✏️ optional: run `predict_ctc` + `error_analysis` for the best CTC model (needs torch) |
| 7 | Ablation study | Results and interpretation | `results/tables/ablation.md`, `docs/RESULTS.md` §7 | ✅ |
| 8 | Technical report | 15-25 pages, section-4 structure | `report/HTR_AI403_Technical_Report_Formatted.pdf` | ✅ 17 pages, 9 figures added (sections 10-14) · original kept in `report/archive/` |
| 9 | Presentation | 15 min, every member presents | `presentation/` (online deck, exportable to PPTX/PDF) | ✅ names and 4-person split in place · ✏️ rehearse timing, re-export the PPTX |
| 10 | Live demonstration | 5 min, live | `demo/app.py` | ✏️ download TrOCR checkpoint, record backup video, rehearse |
| 11 | README | Full run instructions | `README.md` | ✅ |
| 12 | Contribution log + peer evaluation | One per student | `contributions/` | ✏️ logs in place for all 4 members, each member enters their own hours · ⚠️ peer evaluations were removed from the repo but are still required: submit them the way the instructor asks |

## Presentation timing (15 min)

| Part | Slides | Presenter | Time |
|---|---|---|---|
| Problem, hypotheses, dataset, preprocessing, architectures | 1-5 | Saeed Sherif Mohamed | 4:00 |
| Training protocol, comparison, hypothesis verdicts | 6, 10, 11 | Mahmoud Mohamed | 3:30 |
| Attention ablation, learning-rate study, error analysis | 7, 8, 12 | Mohamed Elsaeed Salah | 4:00 |
| TrOCR fine-tuning, conclusion | 9, 14 | Nardeen Samy | 2:30 |
| Questions | 15 | All | 1:00 |
| *(Live demo, slide 13 - separate 5 min)* | 13 | Mohamed Elsaeed Salah + Nardeen Samy | 5:00 |

## Known data-integrity points (be ready if asked)

- `baseline_seed1` was trained with `--seed 2` (same seed as `baseline_seed2`, different results because of GPU non-determinism). It does not change H1.
- The main GRU/BiLSTM runs (20 epochs, batch 8) and attention runs (10 epochs, batch 4) use different budgets; the matched comparison is the ablation.
- TrOCR is a single run.
