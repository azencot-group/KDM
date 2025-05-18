import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import zscore
from sklearn.cluster import KMeans

from data.data_loading.data_loaders import load_data
from utils.names import Datasets

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


def detect_outliers(points, eps=0.5, min_samples=5, z_thresh=3.0):
    """
    Detects outliers in a 2D dataset using DBSCAN and Z-score filtering.

    :param points: (N,2) numpy array of points
    :param eps: DBSCAN epsilon (radius for neighborhood)
    :param min_samples: DBSCAN minimum samples for a core point
    :param z_thresh: Z-score threshold for outlier detection
    :return: inliers (points, indices), outliers (points, indices)
    """

    # **Step 1: DBSCAN Clustering**
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(points)

    # **Step 2: Separate Inliers and Outliers from DBSCAN**
    core_mask = labels != -1  # DBSCAN clusters (-1 are outliers)
    dbscan_outliers = np.where(~core_mask)[0]  # Get outlier indices from DBSCAN

    # **Step 3: Z-Score Outlier Detection**
    z_scores = np.abs(zscore(points, axis=0))  # Compute z-score along x, y
    z_mask = (z_scores > z_thresh).any(axis=1)  # Mark points exceeding threshold
    z_outliers = np.where(z_mask)[0]  # Get outlier indices from Z-score

    # **Step 4: Merge Outliers from Both Methods**
    outlier_indices = np.unique(np.concatenate([dbscan_outliers, z_outliers]))
    inlier_indices = np.setdiff1d(np.arange(len(points)), outlier_indices)

    inliers = points[inlier_indices]
    outliers = points[outlier_indices]

    return (inliers, inlier_indices), (outliers, outlier_indices)


if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

train_data, _ = load_data(
    dataset=Datasets.Checkerboard,
    dataset_path='../../data/checkerboard/sampled_dataset.npy',
    dataset_path_test='',
    batch_size=10000,
    num_workers=4,
)

batch = next(iter(train_data))

plt.scatter(batch[0][:, 0], batch[0][:, 1], c="blue", alpha=0.6)
plt.legend()
plt.title("Dataset Sampled by a Diffusion Model Visualization")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

# Load the model and sample
km = torch.load('model.pt')
samples = km.sample(10000, device)
x0 = samples[0].cpu().detach()
x0_data = batch[0].cpu().detach()
xT = samples[1].cpu().detach()
xT_data = batch[1].cpu().detach()

# === Outliers + clustering for x0_data ===
(inliers_data, inlier_indices_data), (detected_outliers_data, outlier_indices_data) = detect_outliers(x0_data, eps=0.15)
(inliers_data, inlier_indices_data), (detected_outliers_data, outlier_indices_data) = detect_outliers(x0_data, eps=0.15)
kmeans_data = KMeans(n_clusters=8, random_state=42)
labels_data = kmeans_data.fit_predict(x0_data)
labels_inliers_data = labels_data[inlier_indices_data]
labels_outliers_data = labels_data[outlier_indices_data]

# === Outliers + clustering for x0 ===
(inliers, inlier_indices), (detected_outliers, outlier_indices) = detect_outliers(x0, eps=0.15)
kmeans = KMeans(n_clusters=8, random_state=42)
labels = kmeans.fit_predict(x0)
labels_inliers = labels[inlier_indices]
labels_outliers = labels[outlier_indices]

# === Create 2x2 subplot ===
fig, axes = plt.subplots(2, 2, figsize=(10, 10))


def plot(ax, points, in_idx, out_idx, labels_in, title):
    ax.scatter(points[in_idx, 0], points[in_idx, 1], c=labels_in, alpha=0.4, rasterized=True)
    ax.scatter(points[out_idx, 0], points[out_idx, 1], c='red', marker='x', alpha=0.9, rasterized=True)
    ax.set_title(title)
    ax.axis('off')


# First row: generated x₀ and x_T
plot(axes[0, 0], x0, inlier_indices, outlier_indices, labels_inliers, "x₀: Generated")
plot(axes[0, 1], xT, inlier_indices, outlier_indices, labels_inliers, "x_T: Generated")

# Second row: real x₀ and x_T
plot(axes[1, 0], x0_data, inlier_indices_data, outlier_indices_data, labels_inliers_data, "x₀: Real")
plot(axes[1, 1], xT_data, inlier_indices_data, outlier_indices_data, labels_inliers_data, "x_T: Real")

plt.tight_layout()
plt.savefig('outlier_detection.pdf', dpi=300, bbox_inches='tight')
plt.show()
