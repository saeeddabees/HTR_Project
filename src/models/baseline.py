import torch
import torch.nn as nn

from src.datasets.iam import get_loaders
from src.ctc_decode import greedy_decode
from src.evaluate import corpus_cer


class BaselineCNNGRU(nn.Module):
    def __init__(self, num_classes, hidden=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),                      # H: 64 -> 32, W: W -> W/2
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),                      # H: 32 -> 16, W: W/2 -> W/4
        )
        self.rnn = nn.GRU(64 * 16, hidden, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        f = self.cnn(x)                              # (N, 64, 16, W/4)
        n, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(n, w, c * h)  # (N, W/4, 1024)
        out, _ = self.rnn(f)                         # (N, W/4, hidden)
        out = self.fc(out)                           # (N, W/4, num_classes)
        return out.log_softmax(2).permute(1, 0, 2)   # (T, N, C) as CTCLoss expects


def output_lengths(widths):
    return widths // 4  # two poolings of 2


if __name__ == "__main__":
    # Sanity check: try to overfit ONE small batch. If the loss goes down,
    # the whole pipeline (data -> model -> CTC loss -> decoder) works.
    torch.manual_seed(0)
    loaders, char2idx, idx2char = get_loaders(batch_size=4)
    num_classes = len(char2idx) + 1  # +1 for blank
    model = BaselineCNNGRU(num_classes)
    print("Parameters:", sum(p.numel() for p in model.parameters()))

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    imgs, targets, widths, tlens, texts = next(iter(loaders["train"]))
    in_lens = output_lengths(widths)

    for step in range(1, 101):
        optimizer.zero_grad()
        log_probs = model(imgs)
        loss = criterion(log_probs, targets, in_lens, tlens)
        loss.backward()
        optimizer.step()
        if step % 10 == 0:
            print(f"step {step:3d}  loss {loss.item():.3f}")

    model.eval()
    with torch.no_grad():
        preds = greedy_decode(model(imgs), in_lens, idx2char)
    for t, p in zip(texts, preds):
        print("TRUE:", t)
        print("PRED:", p)
    print("CER:", corpus_cer(texts, preds))