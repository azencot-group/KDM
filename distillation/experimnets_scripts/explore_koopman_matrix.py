import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

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
# sort eigvectors by the real value of the eigenvalues
idx = np.argsort(eig_val.real)[::-1]
eig_val = eig_val[idx]
eig_vec = eig_vec[:, idx]

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


# --- TSNE of the Koopman space encodings --- #
t = torch.ones((points.shape[0],)).to(points.device)
zT = km.encoder_xT(torch.tensor(points).cuda().float(), t.cuda())

# tsne plot of zT into 2 dimensions
from sklearn.manifold import TSNE
# Convert zT (Koopman encodings) to NumPy for t-SNE
zT_np = zT.detach().cpu().numpy()[:3000]  # Move to CPU and convert to NumPy
zT_label = labels[:3000]
# Perform t-SNE to reduce to 2 dimensions
tsne = PCA(n_components=2, random_state=42)
zT_tsne = tsne.fit_transform(zT_np)

# Plot the t-SNE results
plt.figure(figsize=(8, 8))
for cluster_id in range(n_clusters):
    cluster_points = zT_np[zT_label == cluster_id]
    x = cluster_points[:, 0][cluster_points[:, 0] < 1]
    y = cluster_points[:, 1][cluster_points[:, 0] < 1]
    plt.scatter(x, y, label=f'Square {cluster_id + 1}', alpha=0.6)

# Formatting the plot
plt.legend()
plt.title("Points Grouped into Squares")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

# --- sparsity measure --- #
print(f'out of {zT_np.shape[1]} coordinates')
print(f'average number of unsparse coordinate {(zT_np > 0.000001).sum(axis=1).mean()}')
print(f'standard deviation of the number of unsparse coordinate {(zT_np > 0.000001).sum(axis=1).std()}')
print(f'maximum of the number of unsparse coordinate {(zT_np > 0.000001).sum(axis=1).max()}')
print(f'minimum of the number of unsparse coordinate {(zT_np > 0.000001).sum(axis=1).min()}')