import torch
from scipy.stats import wasserstein_distance


def measure_distribution_distance(A, B, metric="wasserstein"):
    """
    Measures the distance between two arbitrary distributions.

    Args:
        A (torch.Tensor): Tensor of shape (b, 2), representing the first distribution.
        B (torch.Tensor): Tensor of shape (b, 2), representing the second distribution.
        metric (str): The distance metric to use. Options: "wasserstein", "euclidean", "mmd", or "kl".

    Returns:
        float: The distance between the two distributions.
    """
    if metric == "wasserstein":
        # Compute the Wasserstein distance for each dimension and average them
        dist_x = wasserstein_distance(A[:, 0].cpu().numpy(), B[:, 0].cpu().numpy())
        dist_y = wasserstein_distance(A[:, 1].cpu().numpy(), B[:, 1].cpu().numpy())
        return (dist_x + dist_y) / 2
    elif metric == "euclidean":
        # Compute the average Euclidean distance between all points
        distances = torch.cdist(A, B, p=2)  # Pairwise Euclidean distances
        return distances.mean().item()
    elif metric == "mmd":
        # Maximum Mean Discrepancy (MMD) with Gaussian kernel
        sigma = 1.0
        k_aa = torch.exp(-torch.cdist(A, A, p=2) ** 2 / (2 * sigma ** 2)).mean()
        k_bb = torch.exp(-torch.cdist(B, B, p=2) ** 2 / (2 * sigma ** 2)).mean()
        k_ab = torch.exp(-torch.cdist(A, B, p=2) ** 2 / (2 * sigma ** 2)).mean()
        return (k_aa + k_bb - 2 * k_ab).item()
    elif metric == "kl":
        # Kullback-Leibler (KL) Divergence
        bins = 50
        hist_a, _ = torch.histogramdd(A, bins=(bins, bins), density=True)
        hist_b, _ = torch.histogramdd(B, bins=(bins, bins), density=True)
        hist_a += 1e-10  # Add small value to avoid division by zero
        hist_b += 1e-10
        kl = torch.sum(hist_a * torch.log(hist_a / hist_b))
        return kl.item()
    else:
        raise ValueError(f"Unsupported metric '{metric}'. Use 'wasserstein', 'euclidean', 'mmd', or 'kl'.")


# # Example usage
# A = torch.rand((100, 2))  # Sample distribution A
# B = torch.rand((100, 2))  # Sample distribution B
#
# distance = measure_distribution_distance(A, B, metric="wasserstein")
# print("Wasserstein Distance:", distance)
#
# distance = measure_distribution_distance(A, B, metric="euclidean")
# print("Euclidean Distance:", distance)
#
# distance = measure_distribution_distance(A, B, metric="mmd")
# print("MMD Distance:", distance)
#
# distance = measure_distribution_distance(A, B, metric="kl")
# print("KL Divergence:", distance)
