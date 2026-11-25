
import argparse
import json
from pathlib import Path

import torch

from src.datasets.iam import get_loaders
from src.ctc_decode import greedy_decode
from src.evaluate import corpus_cer, edit_distance

from src.models.baseline import (
    BaselineCNNGRU,
    output_lengths as baseline_output_lengths,
)

from src.models.crnn import (
    CRNNBiLSTM,
    output_lengths as bilstm_output_lengths,
)

from src.models.attention import (
    CRNNBiLSTMAttention,
    output_lengths as attention_output_lengths,
)


MODELS = {
    "baseline": {
        "class": BaselineCNNGRU,
        "output_lengths": baseline_output_lengths,
    },
    "bilstm": {
        "class": CRNNBiLSTM,
        "output_lengths": bilstm_output_lengths,
    },
    "attention": {
        "class": CRNNBiLSTMAttention,
        "output_lengths": attention_output_lengths,
    },
    "attention_aug": {
        "class": CRNNBiLSTMAttention,
        "output_lengths": attention_output_lengths,
    },
}


def evaluate_model(model, loader, idx2char, out_len_fn, device):
    truths = []
    predictions = []

    model.eval()

    with torch.no_grad():
        for imgs, targets, widths, tlens, texts in loader:
            log_probs = model(imgs.to(device)).cpu()

            batch_preds = greedy_decode(
                log_probs,
                out_len_fn(widths),
                idx2char,
            )

            predictions.extend(batch_preds)
            truths.extend(texts)

    cer = corpus_cer(truths, predictions)

    word_errors = sum(
        edit_distance(t.split(), p.split())
        for t, p in zip(truths, predictions)
    )

    total_words = sum(len(t.split()) for t in truths)
    wer = word_errors / total_words if total_words else 0.0

    return cer, wer, truths, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate HTR models on IAM test set"
    )

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        choices=[0, 1, 2],
        required=True,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Model:", args.model)
    print("Seed:", args.seed)

    config = MODELS[args.model]

    checkpoint_path = Path(
        f"checkpoints/{args.model}_seed{args.seed}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # Test set is always evaluated without augmentation.
    loaders, _, _ = get_loaders(
        batch_size=args.batch_size
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    char2idx = checkpoint["char2idx"]
    idx2char = {i: c for c, i in char2idx.items()}

    model = config["class"](
        len(char2idx) + 1
    ).to(device)

    model.load_state_dict(checkpoint["model"])

    print("Checkpoint loaded:", checkpoint_path)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    print("\nEvaluating on IAM test set...")

    cer, wer, truths, predictions = evaluate_model(
        model,
        loaders["test"],
        idx2char,
        config["output_lengths"],
        device,
    )

    print("\n========== TEST RESULTS ==========")
    print("Model:", args.model)
    print("Seed:", args.seed)
    print("Test samples:", len(truths))
    print(f"Test CER: {cer:.4f} ({cer * 100:.2f}%)")
    print(f"Test WER: {wer:.4f} ({wer * 100:.2f}%)")

    results = {
        "model": args.model,
        "seed": args.seed,
        "checkpoint": str(checkpoint_path),
        "test_samples": len(truths),
        "test_cer": cer,
        "test_wer": wer,
    }

    output_dir = Path("experiments")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (
        f"{args.model}_seed{args.seed}_test_results.json"
    )

    if output_path.exists():
        raise FileExistsError(
            f"Results already exist: {output_path}. "
            "Move or rename the old file before rerunning."
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
