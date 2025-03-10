from koopman_distillation.data.data_loading.datasets_objects import CheckerboardDataset, Cifar10Dataset, \
    FIDCifar10Dataset, Cifar10DatasetFastLoading
from koopman_distillation.utils.names import Datasets
import torch


def load_data(dataset: Datasets, dataset_path: str, batch_size: int, num_workers: int):
    if dataset == Datasets.Checkerboard:
        return torch.utils.data.DataLoader(CheckerboardDataset(dataset_path),
                                           num_workers=num_workers,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)


    elif dataset == Datasets.Cifar10:
        return torch.utils.data.DataLoader(Cifar10Dataset(dataset_path),
                                           num_workers=num_workers,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)

    elif dataset == Datasets.Cifar10FastOneStepLoading:
        return torch.utils.data.DataLoader(Cifar10DatasetFastLoading(dataset_path),
                                           num_workers=num_workers,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           drop_last=True)

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


def load_data_for_testing(path):
    return FIDCifar10Dataset(path)
