# Reproduce every model, test result, table, graph and error analysis.
# Run from the project root:  powershell -ExecutionPolicy Bypass -File scripts\reproduce_all.ps1
# Optional:  -SkipTraining  (only evaluate existing checkpoints and rebuild results)
param([switch]$SkipTraining)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Run($cmd) { Write-Host "`n>> python $cmd" -ForegroundColor Cyan; & python @($cmd -split '\s+'); if ($LASTEXITCODE) { throw "failed: $cmd" } }

$seeds = 0, 1, 2
if (-not $SkipTraining) {
    foreach ($s in $seeds) {
        Run "-m src.train --model baseline  --epochs 20 --batch_size 8 --seed $s --name baseline_seed$s"
        Run "-m src.train --model bilstm    --epochs 20 --batch_size 8 --seed $s --name bilstm_seed$s"
        Run "-m src.train --model attention --epochs 10 --batch_size 4 --seed $s --name attention_seed$s"
        Run "-m src.train --model attention --epochs 10 --batch_size 4 --seed $s --augment --name attention_aug_seed$s"
        Run "-m src.train --model bilstm    --epochs 10 --batch_size 4 --seed $s --name ablation_bilstm_seed$s"
    }
    Run "-m src.hyperparameter_search"
    Run "trocr_train.py"
}

# Test evaluation (src.test refuses to overwrite: archive old results first)
New-Item -ItemType Directory -Force experiments\_old_test_results | Out-Null
foreach ($m in "baseline", "bilstm", "attention", "attention_aug") {
    foreach ($s in $seeds) {
        $f = "experiments\${m}_seed${s}_test_results.json"
        if (Test-Path $f) { Move-Item $f experiments\_old_test_results\ -Force }
        Run "-m src.test --model $m --seed $s"
    }
}
Run "-m src.aggregate_results"

# Tables, graphs and error analysis
Run "-m src.analysis.make_results"
Run "-m src.analysis.error_analysis --predictions experiments/trocr_test_predictions.json --name trocr --label TrOCR"
Run "-m src.analysis.predict_ctc --checkpoint checkpoints/attention_aug_seed1.pt --model attention"
Run "-m src.analysis.error_analysis --predictions experiments/attention_aug_seed1_test_predictions.json --name attention_aug_seed1 --label Attention+Aug"
Write-Host "`nAll done. See results/ and docs/RESULTS.md" -ForegroundColor Green
