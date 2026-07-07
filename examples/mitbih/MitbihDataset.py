import torch
from torch.utils.data import Dataset


class MitbihDataset(Dataset):
    """Dataset for MIT-BIH data with RR intervals features.

    Expects an X in (C, T) format 4 RR feature.
    Returns (X_deriv, RR, y) tuples.
    """

    # TODO: add attribute to control whether to return deriv or X or both
    def __init__(self, X, RR, y, get_deriv=True):
        super().__init__()
        self.x = torch.as_tensor(X, dtype=torch.float32)
        self.rr = torch.as_tensor(RR, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)
        self.deriv_x = torch.diff(self.x, dim=-1, prepend=self.x[..., :1])
        self.get_deriv = get_deriv

    def __getitem__(self, index):
        return (
            self.deriv_x[index] if self.get_deriv else self.x[index],
            self.rr[index],
            self.y[index],
        )

    def __len__(self):
        return len(self.y)