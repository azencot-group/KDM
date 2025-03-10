import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.stats import zscore

from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.utils.names import Datasets
from old.distillation.koopman_model import OneStepKoopmanModel


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


train_data = load_data(
    dataset=Datasets.Checkerboard,
    dataset_path='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/checkerboard/sol.npy',
    batch_size=10000,
    num_workers=4,
)

batch = next(iter(train_data))[0]

plt.scatter(batch[:, 0], batch[:, 1], c="blue", alpha=0.6)
plt.legend()
plt.title("Outlier Detection in 2D Plane")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()


# training arguments
lr = 0.0003
batch_size = 4096
iterations = 1301
print_every = 50
time_steps = 10
hidden_dim = 256
noisy_latent = 0.4
rec_xT_loss = False
# push_list = ['all_linear', 'sample_linear', 'batch_linear']
push = 'all_linear'
km = OneStepKoopmanModel(hidden_dim=hidden_dim,
                         time_steps=time_steps,
                         noisy_latent=noisy_latent,
                         push=push,
                         rec_xT_loss=rec_xT_loss).to(device)

km.load_state_dict(torch.load(f'/home/bermann/functional_mapping/distillation/koopman_model_v0.pt'))
samples = km.sample(10000, device)
x0 = samples[0].cpu().detach()
xT = samples[1].cpu().detach()
plt.scatter(x0[:, 0], x0[:, 1])
plt.show()
plt.scatter(xT[:, 0], xT[:, 1])
plt.show()

(inliers, inlier_indices), (detected_outliers, outlier_indices) = detect_outliers(x0, eps=0.15)

plt.scatter(inliers[:, 0], inliers[:, 1], c="blue", label="Inliers")
plt.scatter(detected_outliers[:, 0], detected_outliers[:, 1], c="red", label="Outliers", marker="x")
plt.legend()
plt.title("Outlier Detection in 2D Plane")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()

plt.scatter(xT[inlier_indices, 0], xT[inlier_indices, 1], c="blue", label="Inliers")
plt.scatter(xT[outlier_indices, 0], xT[outlier_indices, 1], c="red", label="Outliers", marker="x")
plt.legend()
plt.title("Outlier Detection in 2D Plane in the noise domain")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.show()




