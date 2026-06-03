import torch
from torch import nn


class CNN(nn.Module):
    """simple CNN with a multihead attention layer for the MIT-BIH arrhythmia detection task

    Inputs:
        X (B, 1, window_len)
        RR (B, 4)
    Outputs: (B, 3) logits
    """

    def __init__(self):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=64, num_heads=4, dropout=0.2, batch_first=True
        )
        self.norm = nn.LayerNorm(64)

        self.linears = nn.Sequential(
            nn.Linear(64 * 8 + 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X, rr):
        out = self.convolutions(X)
        out = out.permute(0, 2, 1)
        attn_out, _ = self.attention(out, out, out)
        out = self.norm(out + attn_out)
        out = torch.cat([out.flatten(1), rr], dim=1)
        return self.linears(out)
