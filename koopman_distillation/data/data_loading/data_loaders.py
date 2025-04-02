import numpy as np
from torch.utils.data import Sampler

from koopman_distillation.data.data_loading.datasets_objects import CheckerboardDataset, Cifar10Dataset, \
    FIDCifar10Dataset, Cifar10DatasetFastLoading
from koopman_distillation.utils.names import Datasets
import torch


class InfiniteBatchSampler(Sampler):
    def __init__(self, dataset_len, batch_size, seed=0, filling=False, shuffle=True, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = dataset_len // batch_size if drop_last else (dataset_len + batch_size - 1) // batch_size
        self.max_p = self.iters_per_ep * batch_size
        self.filling = filling
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed
        self.indices = self.gener_indices()

    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.seed)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.filling:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())

    def __iter__(self):
        self.epoch = 0
        while True:
            self.epoch += 1
            p, q = 0, 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep


def load_data(dataset: Datasets, dataset_path: str, dataset_path_test: str, batch_size: int, num_workers: int):
    if dataset == Datasets.Checkerboard:
        return torch.utils.data.DataLoader(CheckerboardDataset(dataset_path),
                                           num_workers=num_workers,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True), None


    elif dataset == Datasets.Cifar10:
        train_set = Cifar10Dataset(dataset_path)
        train_data = torch.utils.data.DataLoader(dataset=train_set,
                                                 pin_memory=True,
                                                 batch_sampler=InfiniteBatchSampler(dataset_len=len(train_set),
                                                                                    batch_size=batch_size),
                                                 num_workers=num_workers)
        # train_data = torch.utils.data.DataLoader(Cifar10Dataset(dataset_path),
        #                                          num_workers=num_workers,
        #                                          batch_size=batch_size,
        #                                          shuffle=True,
        #                                          drop_last=True)
        test_data = torch.utils.data.DataLoader(Cifar10Dataset(dataset_path_test),
                                                num_workers=num_workers,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
        return train_data, test_data

    elif dataset == Datasets.Cifar10FastOneStepLoading:

        train_data = torch.utils.data.DataLoader(Cifar10DatasetFastLoading(dataset_path),
                                                 num_workers=num_workers,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 drop_last=True)
        test_data = torch.utils.data.DataLoader(Cifar10Dataset(dataset_path_test),
                                                num_workers=num_workers,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
        return train_data, test_data

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


def load_data_for_testing(path):
    return FIDCifar10Dataset(path)
