import copy

import numpy as np
import torch

from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.evaluation.fid import translate_to_image_format, sample_and_calculate_fid
from koopman_distillation.utils.names import Datasets


def experiment1():
    noise_dataset = train_data.dataset.dataset[:, 1]
    image_dataset = train_data.dataset.dataset[:, 0]
    i = 50
    x_T = noise_dataset[i]
    x_0 = image_dataset[i]
    plt.imshow(x_0.transpose(1, 2, 0))
    plt.show()
    noise_dataset_without_x_T = np.concat([noise_dataset[:i], noise_dataset[i + 1:]])
    image_dataset_without_x_T = np.concat([noise_dataset[:i], noise_dataset[i + 1:]])
    # noise_dataset_without_x_T = noise_dataset
    # image_dataset_without_x_T = image_dataset
    # Flatten all for distance computation
    x_T_flat = x_T.flatten()  # shape (C*H*W,)
    noise_flat = noise_dataset_without_x_T.reshape(noise_dataset_without_x_T.shape[0], -1)  # shape (N-1, C*H*W)
    # Compute Euclidean distances
    distances = np.linalg.norm(noise_flat - x_T_flat, axis=1)
    # Get indices of the 31 closest
    nearest_indices = np.argsort(distances)[:32]
    # Fetch the closest noise samples (in original shape)
    closest_x_0 = image_dataset_without_x_T[nearest_indices]  # shape (32, C, H, W)
    closest_x_T = noise_dataset_without_x_T[nearest_indices]  # shape (32, C, H, W)
    # change to tensor and put on device
    closest_x_0 = torch.tensor(closest_x_0).to(device)
    closest_x_T = torch.tensor(closest_x_T).to(device)
    x_T = torch.tensor(x_T).to(device).unsqueeze(dim=0)
    # create a koopman matrix from them
    T = torch.ones((closest_x_0.shape[0],)).to(x_T.device)
    t = torch.zeros((closest_x_0.shape[0],)).to(x_T.device)
    z_0 = model.x_0_observables_encoder(closest_x_0, t, t)
    z_T_data = model.x_T_observables_encoder(closest_x_T, T, T)
    K = torch.linalg.lstsq(z_T_data.reshape(z_0.shape[0], -1), z_0.reshape(z_0.shape[0], -1)).solution
    z_T = model.x_0_observables_encoder(x_T.repeat(32, 1, 1, 1), t, t)
    zt0_push = (z_T.reshape(z_0.shape[0], -1) @ K).reshape(z_T.shape)
    xt0_push_hat = model.x0_observables_decoder(zt0_push, t, t)
    plt.imshow(translate_to_image_format(xt0_push_hat)[0])
    plt.title("new sample")
    plt.show()


with torch.no_grad():
    model_path = "/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond_dmdkoopman/2025_03_18_07_44_58/model.pt"
    model = torch.load(model_path)
    model.eval()
    train_data, test_data = load_data(
        dataset=Datasets.Cifar10FastOneStepLoading,
        dataset_path='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_dataset.npy',
        dataset_path_test='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_test_data',
        batch_size=32,
        num_workers=6,
    )

    tr_batch = next(iter(train_data))
    # ts_batch = next(iter(test_data))
    device = 'cuda'
    model = model.to(device)

    x = 5
    x_0 = tr_batch[0].to(device)
    x_T_data = tr_batch[1].to(device)
    T = torch.ones((x_0.shape[0],)).to(x_0.device)
    t = torch.zeros((x_0.shape[0],)).to(x_0.device)
    z_0 = model.x_0_observables_encoder(x_0, t, t)
    z_T_data = model.x_T_observables_encoder(x_T_data, T, T)
    K = torch.linalg.lstsq(z_T_data.reshape(z_0.shape[0], -1), z_0.reshape(z_0.shape[0], -1)).solution

    # new sample
    z_T = model.x_T_observables_encoder(torch.randn_like(x_T_data) * 80, T, T)
    # original push
    z_T_original = model.x_T_observables_encoder(x_T_data, T, T)

    zt0_push = (z_T.reshape(z_0.shape[0], -1) @ K).reshape(z_T.shape)
    zt0_push_original = (z_T_original.reshape(z_0.shape[0], -1) @ K).reshape(z_T.shape)

    xt0_push_hat = model.x0_observables_decoder(zt0_push, t, t)
    xt0_push_hat_original = model.x0_observables_decoder(zt0_push_original, t, t)

    import matplotlib.pyplot as plt

    i = 1
    plt.imshow(translate_to_image_format(x_0)[i])
    plt.title("original sample")
    plt.show()

    plt.imshow(translate_to_image_format(xt0_push_hat_original)[i])
    plt.title("original sample pushed")
    plt.show()

    plt.imshow(translate_to_image_format(xt0_push_hat)[i])
    plt.title("new sample")
    plt.show()

    # fid = sample_and_calculate_fid(model=model,
    #                                data_shape=(3, 32, 32),
    #                                num_samples=50000,
    #                                device='cuda',
    #                                batch_size=256,
    #                                epoch=0,
    #                                image_dir='/home/bermann/functional_mapping/koopman_distillation/tmp_do_delete',
    #                                data_loader=copy.deepcopy(train_data))

    # --- Technique #1 - find close by noises --- #
    experiment1()

    # --- Technique #2 - find close by noises --- #
    noise_dataset = train_data.dataset.dataset[:, 1]
    image_dataset = train_data.dataset.dataset[:, 0]
    closest_noises = []
    closest_images = []
    x_epsilons = []
    for i in range(32):
        x_eps = np.random.randn(32, 32, 3) * 80
        x_eps_flatter = x_eps.flatten()
        distances = np.linalg.norm(noise_dataset.reshape(-1, 3072) - x_eps_flatter, axis=1)
        nearest_indices = np.argsort(distances)[:1]
        close_xT = noise_dataset[nearest_indices]
        closes_x0 = image_dataset[nearest_indices]
        closest_noises.append(close_xT)
        closest_images.append(closes_x0)
        x_epsilons.append(x_eps[None, :])

    closest_noises = np.vstack(closest_noises)
    closest_images = np.vstack(closest_images)
    x_epsilons = np.vstack(x_epsilons)

    # change to tensor and put on device
    closest_x_0 = torch.tensor(closest_noises).to(device)
    closest_x_T = torch.tensor(closest_images).to(device)
    x_T = torch.tensor(x_epsilons).to(device)
    # create a koopman matrix from them
    T = torch.ones((closest_x_0.shape[0],)).to(x_T.device)
    t = torch.zeros((closest_x_0.shape[0],)).to(x_T.device)
    z_0 = model.x_0_observables_encoder(closest_x_0, t, t)
    z_T_data = model.x_T_observables_encoder(closest_x_T, T, T)
    K = torch.linalg.lstsq(z_T_data.reshape(z_0.shape[0], -1), z_0.reshape(z_0.shape[0], -1)).solution
    z_T = model.x_0_observables_encoder(x_T, t, t)
    zt0_push = (z_T.reshape(z_0.shape[0], -1) @ K).reshape(z_T.shape)
    xt0_push_hat = model.x0_observables_decoder(zt0_push, t, t)
    plt.imshow(translate_to_image_format(xt0_push_hat)[0])
    plt.title("new sample")
    plt.show()
