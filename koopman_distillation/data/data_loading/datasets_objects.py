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
        x0: Tensor = torch.tensor(dynamics[0]).float()
        xT: Tensor = torch.tensor(dynamics[-1]).float()

        return x0, xT, ix


class Cifar10DatasetFastLoading(torch.utils.data.Dataset):
    """
    We upload the data into the RAM memory to speed up the training process.
    """

    def __init__(self, path):
        # parse all the paths in path
        self.dataset = np.load(path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, ix):
        dynamics = self.dataset[ix]
        x0: Tensor = torch.tensor(dynamics[0]).float()
        xT: Tensor = torch.tensor(dynamics[1]).float()

        return x0, xT, ix


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
