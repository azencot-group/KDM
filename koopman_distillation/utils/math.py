import torch


def psudo_hober_loss(x, y, c):
    return torch.sqrt(((x - y) ** 2) + (c**2)) - c