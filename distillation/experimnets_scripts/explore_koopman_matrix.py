import torch
from matplotlib import pyplot as plt

from distillation.utils.display import plot_spectrum
from koopman_model import OneStepKoopmanModel
import numpy as np

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

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

km.load_state_dict(torch.load(f'../koopman_model_v0.pt'))
K = km.instead_of_matrix.weight.data.cpu().numpy()

# --- plot K --- #
plt.imshow(K)
plt.colorbar()
plt.show()

# --- plot eigenvalues spectrum --- #
plot_spectrum(K)

# --- plot eigenfunctions --- #
eig_val, eig_vec = np.linalg.eig(K)
eig_vec = torch.tensor(eig_vec).to('cuda').float()
Xs, Ys = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-4, 4, 100))
Xs, Ys = torch.tensor(Xs).cuda(), torch.tensor(Ys).cuda()
data = torch.stack([Xs.reshape(-1), Ys.reshape(-1)]).T.float()
lifted_grid = km.encoder_x0(data, t=torch.ones(data.shape[0]).cuda())
lifted_grid_eig = lifted_grid @ eig_vec.T

# Plot all eigenvectors in a single plot
fig, axes = plt.subplots(5, 6, figsize=(15, 12))
axes = axes.ravel()

for i in range(30):
    axes[i].imshow(lifted_grid_eig[:, i].reshape(100, 100).detach().cpu(), cmap='viridis')
    axes[i].set_title(f"Eigenvector {i + 1}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# --- data tracking --- #

x0_sample = km.sample(50000, device)
# plot in colors each of the square
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate a simulated dataset (replace this with your numpy array `points`)
# points = np.load('your_points_file.npy') # Load your numpy array here
points = x0_sample[0].detach().cpu().numpy()  # Simulated for demonstration

# Step 1: Use clustering to identify the distinct squares (assuming 8 squares)
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(points)

# Step 2: Plot the points, color-coded by cluster
points = x0_sample[1].detach().cpu().numpy()
plt.figure(figsize=(8, 8))
for cluster_id in range(n_clusters):
    cluster_points = points[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Square {cluster_id + 1}', alpha=0.6)

# Mark the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=100,
            label='Centers')

# Formatting the plot
plt.legend()
plt.title("Points Grouped into Squares")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

# Step 2: Plot the points, color-coded by cluster
points = x0_sample[0].detach().cpu().numpy()
plt.figure(figsize=(8, 8))
for cluster_id in range(n_clusters):
    cluster_points = points[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Square {cluster_id + 1}', alpha=0.6)

# Mark the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=100,
            label='Centers')

# Formatting the plot
plt.legend()
plt.title("Points Grouped into Squares")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
