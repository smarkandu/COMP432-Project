import torch
import numpy as np
from torch.utils.data import dataloader
from pathlib import Path
from sklearn import preprocessing


# Inspired by https://discuss.pytorch.org/t/how-can-i-make-npz-dataloader/94938/3
class NPZ_DataSet(dataloader.Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        print(self.files[item])
        X = np.load('data.npz')['X']
        X = X[:, :, :, 2]  # Get one channel
        X = X.reshape(-1, 1, 120, 120)
        X = X / 255.0  # Normalize data

        y = np.load('data.npz')['y']
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)

        torch_X = torch.from_numpy(X)
        torch_y = torch.from_numpy(y)
        if self.transform is not None:
            torch_X = self.transform(torch_X)

        return torch_X, torch_y
