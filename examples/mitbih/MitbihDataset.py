from torch.utils.data import Dataset


class MitbihDataset(Dataset):
    """Dataset for MIT-BIH data with RR intervals features.

    Expects an X in (C, T) format 4 RR feature.
    Returns (X, RR, y) tuples.
    """

    def __init__(self, X, RR, y):
        super().__init__()
        self.x = X
        self.rr = RR
        self.y = y

    def __getitem__(self, index):
        return [self.x[index], self.rr[index]], self.y[index]

    def __len__(self):
        return len(self.y)