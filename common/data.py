# common/data.py

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# Preprocessing
# ============================================================


def preprocess_spectra(spectra):
    return np.log(np.maximum(spectra, 0.2))


def normalize(x, axis=None):
    mean = x.mean(axis=axis, keepdims=True)
    std = x.std(axis=axis, keepdims=True)

    x_norm = (x - mean) / std

    return x_norm, mean, std


def denormalize(x_norm, mean, std):
    return x_norm * std + mean


# ============================================================
# Dataset
# ============================================================

class SpectraDataset(Dataset):

    def __init__(self, spectra, labels):
        self.x = torch.tensor(
            spectra,
            dtype=torch.float32
        ).unsqueeze(1)

        self.y = torch.tensor(
            labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ============================================================
# Dataloaders
# ============================================================

def get_dataloaders(
    dataset,
    batch_size=64,
    train_fraction=0.7,
    val_fraction=0.15,
    seed=42,
):
    n_total = len(dataset)

    n_train = int(train_fraction * n_total)
    n_val = int(val_fraction * n_total)
    n_test = n_total - n_train - n_val

    generator = torch.Generator().manual_seed(seed)

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader
