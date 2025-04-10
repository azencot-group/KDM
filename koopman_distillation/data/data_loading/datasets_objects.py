import glob

import numpy as np
import torch
from torch import Tensor


class CheckerboardDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.points = torch.tensor(np.load(path).transpose(1, 0, 2))

    def __len__(self):
        return len(self.points)

    def __getitem__(self, ix):
        x0: Tensor = self.points[ix][-1, :]
        xT: Tensor = self.points[ix][0, :]

        return x0, xT, ix


class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # parse all the paths in path
        self.paths = glob.glob(path + '/*')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        dynamics = np.load(self.paths[ix])['arr_0']
        x0: Tensor = torch.tensor(dynamics[-1]).float()
        xT: Tensor = torch.tensor(dynamics[0]).float()

        return x0, xT, np.nan # dummy label


class Cifar10DatasetCond(torch.utils.data.Dataset):
    def __init__(self, path):
        # parse all the paths in path
        self.paths = glob.glob(path + '/*')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        data = np.load(self.paths[ix])
        dynamics = data['endpoints']
        x0: Tensor = torch.tensor(dynamics[0]).float()
        xT: Tensor = torch.tensor(dynamics[-1]).float()

        label = data['label']

        return x0, xT, label


class FIDCifar10Dataset(torch.utils.data.Dataset):
    def __init__(self, path):
        # parse all the paths in path
        self.paths = glob.glob(path + '/*')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        dynamics = np.load(self.paths[ix])['arr_0']
        if len(dynamics.shape) == 4:
            x0: Tensor = torch.tensor(dynamics[0]).float()
        else:
            x0: Tensor = torch.tensor(dynamics).float()
        return x0
