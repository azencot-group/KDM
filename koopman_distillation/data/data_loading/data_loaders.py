import numpy as np
import torch

from koopman_distillation.data.data_loading.datasets_objects import CheckerboardDataset, Cifar10Dataset, \
    FIDCifar10Dataset, Cifar10DatasetCond
from koopman_distillation.utils.names import Datasets
from koopman_distillation.utils.dist_lib import get_world_size, get_rank


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


def load_data(args, dataset: Datasets, dataset_path: str, dataset_path_test: str, batch_size: int, num_workers: int):
    if dataset == Datasets.Checkerboard:
        return torch.utils.data.DataLoader(CheckerboardDataset(dataset_path),
                                           num_workers=num_workers,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True), None


    elif dataset == Datasets.Cifar10_1M_Uncond:
        train_set = Cifar10Dataset(dataset_path)
        train_data = InfiniteSampler(dataset=train_set, rank=get_rank(), num_replicas=get_world_size(),
                                     seed=(args.seed + get_rank()))
        train_data_iterator = iter(
            torch.utils.data.DataLoader(dataset=train_set, sampler=train_data, batch_size=batch_size,
                                        pin_memory=True))

        test_data = torch.utils.data.DataLoader(Cifar10Dataset(dataset_path_test),
                                                num_workers=num_workers,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
        return train_data_iterator, test_data

    elif dataset == Datasets.Cifar10_1M_Cond:
        train_set = Cifar10DatasetCond(dataset_path)
        train_data = InfiniteSampler(dataset=train_set, rank=get_rank(), num_replicas=get_world_size(),
                                     seed=(args.seed + get_rank()))
        train_data_iterator = iter(
            torch.utils.data.DataLoader(dataset=train_set, sampler=train_data, batch_size=batch_size,
                                        pin_memory=True))

        test_data = torch.utils.data.DataLoader(Cifar10Dataset(dataset_path_test),
                                                num_workers=num_workers,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                drop_last=True)
        return train_data_iterator, test_data

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


def load_data_for_testing(path):
    return FIDCifar10Dataset(path)
