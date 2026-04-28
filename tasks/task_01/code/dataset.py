import numpy as np
import torch
from torch.utils.data import DataLoader, random_split, Dataset


class SpectraDataset(Dataset):
    def __init__(self, spectra, labels):
        # add channel dimension for CNNs
        self.spectra = torch.tensor(spectra, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx]


def get_dataloaders(spectra, labels, batch_size=32, val_split=0.2, test_split=0.1):
    dataset = SpectraDataset(spectra, labels)
    n_samples = len(dataset)
    n_test = int(n_samples * test_split)
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val - n_test

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [n_train, n_val, n_test])
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
