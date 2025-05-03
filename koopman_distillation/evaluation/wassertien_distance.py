import torch
from scipy.stats import wasserstein_distance


def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data.float()


def wess_distance(A, B):
    dist_x = wasserstein_distance(A[:, 0].cpu().numpy(), B[:, 0].cpu().numpy())
    dist_y = wasserstein_distance(A[:, 1].cpu().numpy(), B[:, 1].cpu().numpy())
    return (dist_x + dist_y) / 2


def measure_wess_distance(model, device, train_loader, num_samples=40000):
    # load num_samples from the train_loader
    final_samples = []
    for i, (x0, xT, _) in enumerate(train_loader):
        xT_sample = xT
        x0_sample = x0
        samples = model.sample(num_samples, device, data_shape=(2,), data_batch=[x0_sample, xT_sample])[0]
        final_samples.append(samples)

    samples = torch.cat(final_samples, dim=0)

    original_data = inf_train_gen(num_samples, device=device)

    return wess_distance(samples[:num_samples].cpu().detach(), original_data[:num_samples])
