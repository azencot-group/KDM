import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.utils.names import Datasets
from old.distillation.utils.display import plot_spectrum
from sklearn.cluster import KMeans

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

# --- plot each photo and class name --- #
# import matplotlib.pyplot as plt
#
# for img, cls in zip(batch[0], classes):
#     plt.imshow(img.permute(1, 2, 0).detach().cpu())
#     plt.title(class_to_name[int(cls.item())])
#     plt.show()

model = torch.load(
    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_17_50_26/model.pt')  # sota
# '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_22_31_21/model.pt')  # sota
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

# Assuming you have some embeddings or representations to cluster (e.g., from a model)
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

# class_to_name = {
#     0: 'Airplane',
#     1: 'Automobile',
#     2: 'Bird',
#     3: 'Cat',
#     4: 'Deer',
#     5: 'Dog',
#     6: 'Frog',
#     7: 'Horse',
#     8: 'Ship',
#     9: 'Truck'
# }
#
# K = model.koopman_operator.weight.data.cpu().numpy()
#
# # --- visual reconstruction quality --- #
# img1 = (batch[1] * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
# img2 = (x0_hat * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
# plt.imshow(img1)
# plt.show()
# plt.imshow(img2)
# plt.show()
# plt.imshow(abs(img1 - img2))
# plt.colorbar()
# plt.show()
#
# # # --- plot K --- #
# # plt.imshow(K, interpolation='none', cmap='bwr')
# # plt.spy(K, precision=1e-2)
# # plt.show()
#
# # # --- plot eigenvalues spectrum --- #
# # plot_spectrum(K)
#
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
