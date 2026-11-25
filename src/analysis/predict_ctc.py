"""
Export per-line test predictions for a trained CTC checkpoint, so the CTC
models can be analysed with src/analysis/error_analysis.py exactly like TrOCR.

Requires torch + the IAM-line dataset (downloaded automatically by Hugging Face).

Run from the project root, e.g. for the best custom model:

    python -m src.analysis.predict_ctc --checkpoint checkpoints/attention_aug_seed1.pt --model attention
    python -m src.analysis.error_analysis --predictions experiments/attention_aug_seed1_test_predictions.json --name attention_aug_seed1 --label "Attention + Aug (seed 1)"
"""

import argparse
import json
from pathlib import Path

import torch

from src.datasets.iam import get_loaders
from src.test import MODELS, evaluate_model

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="path to a .pt file in checkpoints/")
    ap.add_argument("--model", required=True, choices=["baseline", "bilstm", "attention"])
    ap.add_argument("--split", default="test", choices=["validation", "test"])
    ap.add_argument("--batch_size", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    char2idx = ckpt["char2idx"]
    idx2char = {i: c for c, i in char2idx.items()}

    cfg = MODELS[args.model]
    model = cfg["class"](len(char2idx) + 1).to(device)
    model.load_state_dict(ckpt["model"])

    loaders, _, _ = get_loaders(batch_size=args.batch_size)
    cer, wer, truths, preds = evaluate_model(model, loaders[args.split], idx2char,
                                             cfg["output_lengths"], device)

    records = [{"index": i, "ground_truth": t, "prediction": p}
               for i, (t, p) in enumerate(zip(truths, preds))]
    out = ROOT / "experiments" / f"{ckpt_path.stem}_{args.split}_predictions.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{ckpt_path.stem}: {args.split} CER {cer * 100:.2f}%  WER {wer * 100:.2f}%  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
