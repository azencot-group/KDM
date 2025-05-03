import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.utils.names import Datasets
from sklearn.cluster import KMeans


# todo - make this script self sustained and clean

cifar10_cls = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
bs = 1024

train_data, test_data = load_data(
    dataset=Datasets.Cifar10_1M_Uncond,
    dataset_path='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond',
    batch_size=bs,
    num_workers=4,
    dataset_path_test='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_test_data',

)

batch = next(iter(train_data))
classes = torch.argmax(cifar10_cls(batch[1]), dim=1)

class_to_name = {
    0: 'Airplane',
    1: 'Automobile',
    2: 'Bird',
    3: 'Cat',
    4: 'Deer',
    5: 'Dog',
    6: 'Frog',
    7: 'Horse',
    8: 'Ship',
    9: 'Truck'
}

model = torch.load(
    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_17_50_26/models.pt')  # sota
# '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_22_31_21/models.pt')  # sota
args = torch.load(
    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_17_50_26/args.pth')  # sote
# '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_22_31_21/args.pth')
print(args.__dict__)
with torch.no_grad():
    model = model.cuda()
    model.eval()
    z0 = model.x_0_observables_encoder(batch[1].cuda(), torch.zeros(batch[0].shape[0]).cuda(),
                                       torch.zeros_like(torch.nn.functional.one_hot(classes).cuda()))
    zT = model.x_T_observables_encoder(batch[0].cuda(), torch.ones(batch[0].shape[0]).cuda(),
                                       torch.zeros_like(torch.nn.functional.one_hot(classes).cuda()))
    x0_hat = model.x0_observables_decoder(z0, torch.ones(batch[0].shape[0]).cuda(),
                                          torch.zeros_like(torch.nn.functional.one_hot(classes).cuda()))

# --- Clustering Evaluation Metrics --- #
# True classes (ground truth)
true_labels = torch.argmax(cifar10_cls(batch[1]), dim=1).cpu().numpy()
# Data to evaluate (e.g., images or embeddings)

# data = zT
# data = batch[1]
# data = z0
data = batch[0]

# normalize data between 1 to -1
data_min = data.min()
data_max = data.max()
data = 2 * (data - data_min) / (data_max - data_min) - 1

# Assuming you have some embeddings or representations to cluster (e.g., from a models)
# Flatten the data for clustering (batch_size x features)
X = data.view(data.size(0), -1).cpu().numpy()
# Run KMeans clustering
kmeans = KMeans(n_clusters=len(np.unique(true_labels)), random_state=42, tol=1e-2).fit(X)
pred_labels = kmeans.labels_
# Baseline: random assignment
random_labels = np.random.permutation(pred_labels.copy())
print("Clustering quality vs. ground truth:")
print(f"  Adjusted Rand Index (ARI): {adjusted_rand_score(true_labels, pred_labels):.4f}")
print(f"  Normalized Mutual Info (NMI): {normalized_mutual_info_score(true_labels, pred_labels):.4f}")
print("Random clustering baseline:")
print(f"  ARI: {adjusted_rand_score(true_labels, random_labels):.4f}")
print(f"  NMI: {normalized_mutual_info_score(true_labels, random_labels):.4f}")

noise = torch.randn_like(batch[0]).view(data.size(0), -1).cpu().numpy()
x0 = batch[1].view(data.size(0), -1).cpu().numpy()
xT = batch[0].view(data.size(0), -1).cpu().numpy()
zT = zT.view(data.size(0), -1).cpu().numpy()
z0 = z0.view(data.size(0), -1).cpu().numpy()
labels = true_labels

xT_0 = xT[labels == 0]
xT_1 = xT[labels == 1]
xT_2 = xT[labels == 2]
xT_3 = xT[labels == 3]
xT_4 = xT[labels == 4]
xT_5 = xT[labels == 5]
xT_6 = xT[labels == 6]
xT_7 = xT[labels == 7]
xT_8 = xT[labels == 8]
xT_9 = xT[labels == 9]

# calculate the average, standard deviation and maximum of the distance between all points between two point groups
from scipy.spatial.distance import cdist
import pandas as pd
groups = [xT_0, xT_1, xT_2, xT_3, xT_4, xT_5, xT_6, xT_7, xT_8, xT_9]
# Initialize empty table
mean_table = pd.DataFrame(index=list(class_to_name.values()), columns=list(class_to_name.values()))

# Fill table
for i in range(10):
    for j in range(10):
        distances = cdist(groups[i], groups[j])
        mean_distance = np.max(distances)
        mean_table.iloc[i, j] = f"{mean_distance:.3f}"
# Print the whole table nicely
print(mean_table.to_string())


from sklearn.metrics import davies_bouldin_score
score_x0 = davies_bouldin_score(x0, labels)
print("DBI score x0: ", score_x0)
score_z0 = davies_bouldin_score(z0, labels)
print("DBI score z0: ", score_z0)

score_xT = davies_bouldin_score(xT, labels)
print("DBI score xT: ", score_xT)
score_zT = davies_bouldin_score(zT, labels)
print("DBI score zT: ", score_zT)

score_noise = davies_bouldin_score(noise, labels)
print("DBI score noise: ", score_noise)

from sklearn.metrics import calinski_harabasz_score
score_x0 = calinski_harabasz_score(x0, labels)
print("Calinski score x0: ", score_x0)
score_z0 = calinski_harabasz_score(z0, labels)
print("Calinski score z0: ", score_z0)

score_xT = calinski_harabasz_score(xT, labels)
print("Calinski score xT: ", score_xT)
score_zT = calinski_harabasz_score(zT, labels)
print("Calinski score zT: ", score_zT)

score_noise = calinski_harabasz_score(noise, labels)
print("Calinski score noise: ", score_noise)

# # --- plot a tsne plot the images on 2D with colors as labels ----
from sklearn.manifold import TSNE

# Reduce dimensions with t-SNE
tsne = PCA(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(batch[0].reshape(bs, -1))

# Plot the t-SNE results
plt.figure(figsize=(8, 8))
for cluster_id in range(10):
    cluster_points = x_tsne[classes == cluster_id]
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]
    # add the name
    plt.scatter(x, y, label=class_to_name[cluster_id], alpha=0.6)

# Formatting the plot
plt.legend()
plt.title("Cifar10 noise to 2D")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(batch[1].reshape(bs, -1))

# Plot the t-SNE results
plt.figure(figsize=(8, 8))
for cluster_id in range(10):
    cluster_points = x_tsne[classes == cluster_id]
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]
    # add the name
    plt.scatter(x, y, label=class_to_name[cluster_id], alpha=0.6)

# Formatting the plot
plt.legend()
plt.title("Cifar10 data to 2D")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(z0.reshape(bs, -1).cpu().detach())
# Plot the t-SNE results

plt.figure(figsize=(8, 8))
for cluster_id in range(10):
    cluster_points = x_tsne[classes == cluster_id]
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]
    # add the name
    plt.scatter(x, y, label=class_to_name[cluster_id], alpha=0.6)
# Formatting the plot
plt.legend()
plt.title("Cifar10 data koopman space to 2D")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()

tsne = TSNE(n_components=2, random_state=42)
x_tsne = tsne.fit_transform(zT.reshape(bs, -1).cpu().detach())
# Plot the t-SNE results

plt.figure(figsize=(8, 8))
for cluster_id in range(10):
    cluster_points = x_tsne[classes == cluster_id]
    x = cluster_points[:, 0]
    y = cluster_points[:, 1]
    # add the name
    plt.scatter(x, y, label=class_to_name[cluster_id], alpha=0.6)
# Formatting the plot
plt.legend()
plt.title("Cifar10 koopman space noise to 2D")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
