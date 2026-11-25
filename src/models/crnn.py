
import torch
import torch.nn as nn


class CRNNBiLSTM(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.rnn = nn.LSTM(
            input_size=64 * 16,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=True
        )

        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        features = self.cnn(x)
        features = features.permute(0, 3, 1, 2)

        batch_size, width, channels, height = features.shape
        features = features.reshape(
            batch_size, width, channels * height
        )

        sequence, _ = self.rnn(features)
        output = self.classifier(sequence)
        output = output.log_softmax(dim=2)

        return output.permute(1, 0, 2)


def output_lengths(widths):
    return widths // 4


if __name__ == "__main__":
    model = CRNNBiLSTM(num_classes=80).cuda()
    dummy_input = torch.randn(2, 1, 64, 400).cuda()

    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
