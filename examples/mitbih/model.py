import torch
from torch import nn


class tinyCNN(nn.Module):
    """Frugal Model based on Matched Filters proposed by (Farag et al., 23)
    for edge ECG-based arrhythmia detection.

    Parameters
    ----------
    matched_filters : Tensor of shape (output_channels, 1, window_len)
        represents the average train signals windowed by subclass. Hence, output_channels
        is the number of subclasses considered. For eg., for MIT-BIH, its equal to 11.

    trainable_conv : Boolean
        whether to train the conv layer initialized with the matched_filters.
    """

    def __init__(self, matched_filters, trainable_conv=True):
        super().__init__()
        mf = torch.as_tensor(matched_filters, dtype=torch.float32)
        if not trainable_conv:
            self.register_buffer("mf", mf)
        else:
            self.mf = nn.Parameter(mf.clone())

        n_filters = mf.shape[0]

        self.post_convolution = nn.Sequential(
            nn.BatchNorm1d(n_filters),
            nn.Tanh(),
            nn.AdaptiveMaxPool1d(1),
        )

        self.rr_path = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

        self.merger = nn.Sequential(nn.Linear(n_filters + 8, 3))

        self._init_weights()

    def _init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, X, rr):
        conv_out = nn.functional.conv1d(
            input=X, weight=self.mf, padding="same"
        )
        conv_out = self.post_convolution(conv_out)
        rr_path_out = self.rr_path(rr)
        return self.merger(
            torch.cat([conv_out.squeeze(-1), rr_path_out], dim=1)
        )


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

    def forward(self, X: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        out = self.convolutions(X)
        out = out.permute(0, 2, 1)
        attn_out, _ = self.attention(out, out, out)
        out = self.norm(out + attn_out)
        out = torch.cat([out.flatten(1), rr], dim=1)
        return self.linears(out)
