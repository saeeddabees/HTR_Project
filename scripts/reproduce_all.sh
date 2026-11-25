#!/usr/bin/env bash
# Reproduce every model, test result, table, graph and error analysis.
# Run from anywhere:  bash scripts/reproduce_all.sh [--skip-training]
set -euo pipefail
cd "$(dirname "$0")/.."
run() { echo; echo ">> python $*"; python "$@"; }

if [[ "${1:-}" != "--skip-training" ]]; then
  for s in 0 1 2; do
    run -m src.train --model baseline  --epochs 20 --batch_size 8 --seed $s --name baseline_seed$s
    run -m src.train --model bilstm    --epochs 20 --batch_size 8 --seed $s --name bilstm_seed$s
    run -m src.train --model attention --epochs 10 --batch_size 4 --seed $s --name attention_seed$s
    run -m src.train --model attention --epochs 10 --batch_size 4 --seed $s --augment --name attention_aug_seed$s
    run -m src.train --model bilstm    --epochs 10 --batch_size 4 --seed $s --name ablation_bilstm_seed$s
  done
  run -m src.hyperparameter_search
  run trocr_train.py
fi

mkdir -p experiments/_old_test_results
for m in baseline bilstm attention attention_aug; do
  for s in 0 1 2; do
    f="experiments/${m}_seed${s}_test_results.json"
    [[ -f "$f" ]] && mv "$f" experiments/_old_test_results/
    run -m src.test --model $m --seed $s
  done
done
run -m src.aggregate_results

run -m src.analysis.make_results
run -m src.analysis.error_analysis --predictions experiments/trocr_test_predictions.json --name trocr --label TrOCR
run -m src.analysis.predict_ctc --checkpoint checkpoints/attention_aug_seed1.pt --model attention
run -m src.analysis.error_analysis --predictions experiments/attention_aug_seed1_test_predictions.json --name attention_aug_seed1 --label "Attention+Aug"
echo; echo "All done. See results/ and docs/RESULTS.md"
