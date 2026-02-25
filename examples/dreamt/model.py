"""Temporary torch model to use. Will encapsulate this latter."""

from torch import nn


class CNN(nn.Module):
    def __init__(self, channel_in=7, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=channel_in,
            out_channels=128,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(32)
        self.bn4 = nn.BatchNorm1d(32)

        self.maxpool = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(
            in_channels=128,
            out_channels=64,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.conv3 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.conv4 = nn.Conv1d(
            in_channels=32,
            out_channels=32,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
        )

        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = x.mean(dim=2)
        x = self.relu(self.fc1(x))
        return self.fc2(x)
