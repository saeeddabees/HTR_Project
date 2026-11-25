
import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.datasets.iam import get_loaders, collate
from src.ctc_decode import greedy_decode
from src.evaluate import corpus_cer, edit_distance

from src.models.baseline import (
    BaselineCNNGRU,
    output_lengths as baseline_output_lengths
)

from src.models.crnn import (
    CRNNBiLSTM,
    output_lengths as bilstm_output_lengths
)

from src.models.attention import (
    CRNNBiLSTMAttention,
    output_lengths as attention_output_lengths
)


# Each entry: (model class, function that computes output lengths)
MODELS = {
    "baseline": (BaselineCNNGRU, baseline_output_lengths),
    "bilstm": (CRNNBiLSTM, bilstm_output_lengths),
    "attention": (CRNNBiLSTMAttention, attention_output_lengths),
}


@torch.no_grad()
def evaluate_loader(model, loader, idx2char, out_len_fn, device):
    model.eval()

    truths, preds = [], []

    for imgs, targets, widths, tlens, texts in loader:
        log_probs = model(imgs.to(device)).cpu()

        preds += greedy_decode(
            log_probs,
            out_len_fn(widths),
            idx2char
        )

        truths += texts

    cer_v = corpus_cer(truths, preds)

    w_err = sum(
        edit_distance(t.split(), p.split())
        for t, p in zip(truths, preds)
    )

    w_tot = sum(len(t.split()) for t in truths)

    return cer_v, w_err / max(w_tot, 1)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        default="baseline",
        choices=list(MODELS)
    )

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument(
        "--subset",
        type=int,
        default=0,
        help="use only N training lines for quick tests"
    )

    ap.add_argument(
        "--name",
        default=None,
        help="experiment name"
    )

    ap.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (Attention model only)"
    )

    args = ap.parse_args()

    # Augmentation is allowed only for the Attention model.
    if args.augment and args.model != "attention":
        ap.error("--augment can only be used with --model attention")

    # Reproducibility
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Model:", args.model)
    print("Augmentation:", "ON" if args.augment else "OFF")

    # Load dataset with the selected augmentation setting.
    loaders, char2idx, idx2char = get_loaders(
        batch_size=args.batch_size,
        augment_train=args.augment
    )

    # Optional subset for quick experiments.
    if args.subset:
        ds = loaders["train"].dataset

        loaders["train"] = DataLoader(
            Subset(ds, range(min(args.subset, len(ds)))),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate
        )

    # Initialize model
    model_cls, out_len_fn = MODELS[args.model]

    model = model_cls(len(char2idx) + 1).to(device)

    n_params = sum(
        p.numel() for p in model.parameters()
    )

    print("Parameters:", n_params)

    # Loss and optimizer
    criterion = nn.CTCLoss(
        blank=0,
        zero_infinity=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr
    )

    # Experiment name
    run_name = args.name or f"{args.model}_seed{args.seed}"

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)

    history = []
    best_cer = float("inf")

    t_start = time.time()

    # Training loop
    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        steps = 0

        for imgs, targets, widths, tlens, texts in loaders["train"]:

            imgs = imgs.to(device)

            optimizer.zero_grad()

            log_probs = model(imgs)

            loss = criterion(
                log_probs,
                targets,
                out_len_fn(widths),
                tlens
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            total_loss += loss.item()
            steps += 1

        # Validation
        val_cer, val_wer = evaluate_loader(
            model,
            loaders["validation"],
            idx2char,
            out_len_fn,
            device
        )

        avg = total_loss / max(steps, 1)

        elapsed = time.time() - t_start

        print(
            f"epoch {epoch:2d} | "
            f"train loss {avg:.3f} | "
            f"val CER {val_cer:.4f} | "
            f"val WER {val_wer:.4f} | "
            f"{elapsed:.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": avg,
            "val_cer": val_cer,
            "val_wer": val_wer
        })

        # Save best checkpoint using validation CER.
        if val_cer < best_cer:

            best_cer = val_cer

            torch.save(
                {
                    "model": model.state_dict(),
                    "char2idx": char2idx,
                    "args": vars(args)
                },
                f"checkpoints/{run_name}.pt"
            )

            print("Best checkpoint saved.")

    # Save experiment results
    log = {
        "run": run_name,
        "args": vars(args),
        "augmentation_enabled": args.augment,
        "params": n_params,
        "train_time_sec": time.time() - t_start,
        "device": device,
        "best_val_cer": best_cer,
        "history": history
    }

    with open(f"experiments/{run_name}.json", "w") as f:
        json.dump(log, f, indent=2)

    print("Done. Best val CER:", best_cer)


if __name__ == "__main__":
    main()
