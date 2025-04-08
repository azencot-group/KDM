import torch
import torch.nn.functional as F


def psudo_hober_loss(x, y, c):
    return torch.sqrt(((x - y) ** 2) + (c ** 2)) - c


def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss
