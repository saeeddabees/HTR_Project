
import json
import statistics
from pathlib import Path


# Project paths
ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT / "experiments"


# All models use consistent seed-based filenames
MODELS = {
    "Baseline CNN-GRU": {
        "files": [
            "baseline_seed0_test_results.json",
            "baseline_seed1_test_results.json",
            "baseline_seed2_test_results.json",
        ],
    },
    "CNN-BiLSTM": {
        "files": [
            "bilstm_seed0_test_results.json",
            "bilstm_seed1_test_results.json",
            "bilstm_seed2_test_results.json",
        ],
    },
    "CNN-BiLSTM-Attention": {
        "files": [
            "attention_seed0_test_results.json",
            "attention_seed1_test_results.json",
            "attention_seed2_test_results.json",
        ],
    },
    "Attention + Augmentation": {
        "files": [
            "attention_aug_seed0_test_results.json",
            "attention_aug_seed1_test_results.json",
            "attention_aug_seed2_test_results.json",
        ],
    },
}


# TrOCR official test results
TROCR_RESULTS = {
    "test_cer": 7.0167,
    "test_wer": 14.8664,
    "test_samples": 2915,
    "number_of_runs": 1,
}


def load_result(filename):
    path = EXPERIMENTS / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing result file: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_metric(data, metric):
    """
    Extract CER or WER from JSON.
    Supports values stored as ratios or percentages.
    """

    possible_keys = {
        "cer": ["test_cer", "cer", "Test CER"],
        "wer": ["test_wer", "wer", "Test WER"],
    }

    for key in possible_keys[metric]:
        if key in data:
            value = float(data[key])

            if value <= 1:
                value *= 100

            return value

    raise KeyError(
        f"Could not find {metric.upper()} in JSON. "
        f"Available keys: {list(data.keys())}"
    )


def main():

    print("=" * 80)
    print("FINAL TEST RESULTS AGGREGATION")
    print("=" * 80)

    summary = {}

    # Process models with three seeds
    for model_name, config in MODELS.items():

        results = []

        print(f"\n{model_name}")
        print("-" * 80)

        for seed, filename in enumerate(config["files"]):

            data = load_result(filename)

            cer = get_metric(data, "cer")
            wer = get_metric(data, "wer")

            results.append({
                "seed": seed,
                "file": filename,
                "cer": cer,
                "wer": wer,
            })

            print(
                f"Seed {seed}: "
                f"CER = {cer:.2f}% | "
                f"WER = {wer:.2f}%"
            )

        # Calculate mean and sample standard deviation
        cer_values = [r["cer"] for r in results]
        wer_values = [r["wer"] for r in results]

        mean_cer = statistics.mean(cer_values)
        mean_wer = statistics.mean(wer_values)

        std_cer = statistics.stdev(cer_values)
        std_wer = statistics.stdev(wer_values)

        # Best seed based on CER
        best = min(results, key=lambda r: r["cer"])

        summary[model_name] = {
            "seeds": results,
            "mean_cer": mean_cer,
            "std_cer": std_cer,
            "mean_wer": mean_wer,
            "std_wer": std_wer,
            "best_seed": best["seed"],
            "best_seed_cer": best["cer"],
            "best_seed_wer": best["wer"],
        }

        print("\nAggregated Results:")
        print(f"Mean CER: {mean_cer:.2f}%")
        print(f"Std CER:  {std_cer:.2f}%")
        print(f"Mean WER: {mean_wer:.2f}%")
        print(f"Std WER:  {std_wer:.2f}%")

        print(
            f"Best Seed: {best['seed']} "
            f"(CER = {best['cer']:.2f}%)"
        )

    # Add TrOCR as a single-run result
    summary["TrOCR"] = TROCR_RESULTS

    # Final comparison table
    print("\n" + "=" * 80)
    print("FINAL MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':<32}"
        f"{'Test CER':>23}"
        f"{'Test WER':>23}"
    )

    print("-" * 80)

    for model_name, result in summary.items():

        if "mean_cer" in result:

            cer_text = (
                f"{result['mean_cer']:.2f} "
                f"+/- {result['std_cer']:.2f}%"
            )

            wer_text = (
                f"{result['mean_wer']:.2f} "
                f"+/- {result['std_wer']:.2f}%"
            )

        else:

            cer_text = (
                f"{result['test_cer']:.2f}% (single)"
            )

            wer_text = (
                f"{result['test_wer']:.2f}% (single)"
            )

        print(
            f"{model_name:<32}"
            f"{cer_text:>23}"
            f"{wer_text:>23}"
        )

    # Save summary JSON
    output_path = EXPERIMENTS / "final_test_summary.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 80)
    print("Aggregation completed successfully!")
    print(f"Summary saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
