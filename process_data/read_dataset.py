import pandas as pd
import numpy as np
import os
import h5py
import torch.utils.data as data

class DATASET(data.Dataset):
    def __init__(self, root, train):
        f = h5py.File(root, 'r')
        if train:
            self.X = f['X_tr'][:]
            self.Y = f['Y_tr'][:]
        else:
            self.X = f['X_vl'][:]
            self.Y = f['Y_vl'][:]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, item):
        return self.X[item], self.Y[item]

