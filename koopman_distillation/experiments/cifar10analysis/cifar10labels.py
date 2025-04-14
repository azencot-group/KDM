import numpy as np
import torch
import matplotlib.pyplot as plt
from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.utils.names import Datasets
from old.distillation.utils.display import plot_spectrum

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
    # '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_17_50_26/model.pt') # sota
    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_22_31_21/model.pt')  # sota
args = torch.load(
    # '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_17_50_26/args.pth') # sote
    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_08_22_31_21/args.pth')
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

K = model.koopman_operator.weight.data.cpu().numpy()

# --- visual reconstruction quality --- #
img1 = (batch[1] * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
img2 = (x0_hat * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()[0]
plt.imshow(img1)
plt.show()
plt.imshow(img2)
plt.show()
plt.imshow(abs(img1 - img2))
plt.colorbar()
plt.show()

# # --- plot K --- #
# plt.imshow(K, interpolation='none', cmap='bwr')
# plt.spy(K, precision=1e-2)
# plt.show()

# # --- plot eigenvalues spectrum --- #
# plot_spectrum(K)

# --- plot a tsne plot the images on 2D with colors as labels ----
from sklearn.manifold import TSNE

# Reduce dimensions with t-SNE
tsne = TSNE(n_components=2, random_state=42)
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
