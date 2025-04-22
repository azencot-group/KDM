import copy
import os
import pickle

import numpy as np
import torch
from pytorch_image_generation_metrics import get_inception_score_and_fid
from tqdm import tqdm
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

# from edm.dnnlib.util import open_url

# with open_url('https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz') as f:
#     ref = dict(np.load(f))
#
# np.save(
#     '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_14_17_44_10/cifar10-32x32',
#     ref)
# to ignore paths
to_igonre = ['2025_04_08_17_16_28', '2025_04_07_21_31_03', '2025_04_08_17_19_43', '2025_04_08_17_21_25',
             '2025_04_08_17_25_25', '2025_04_07_16_25_42', '2025_04_08_22_38_29',
             '2025_04_08_17_55_15', '2025_04_08_17_25_32', '2025_04_08_19_29_33', '2025_04_08_22_41_18',
             '2025_04_08_22_40_35', '2025_04_04_16_51_44', '2025_04_14_16_32_56', '2025_04_14_17_44_10',
             '2025_04_14_16_23_44']
base_path = '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/'
# load all sub directoiries
models_paths = os.listdir(base_path)
# filter out the ones that are not directories
models_paths = [os.path.join(base_path, path) for path in models_paths if os.path.isdir(os.path.join(base_path, path))]
# filter out ones without model.pt file
models_paths = [path for path in models_paths if os.path.isfile(os.path.join(path, 'model.pt'))]
# iterate over the direcotres
for path in models_paths:
    print(path)
    # check first if cond_type exists in args
    args = torch.load(f'{path}/args.pth')

    if any(x in path for x in to_igonre):
        continue
    if hasattr(args, 'cond_type'):
        if args.cond_type == "Uncond":
            continue

    model = torch.load(f'{path}/model.pt')

    model.eval()

    # ---- calculate FID for different noise levels ---- #
    i = 0
    cond = args.cond_type != "Uncond"
    device = 'cuda'
    batch_size = 128
    data_shape = (3, 32, 32)
    sample_noise_z_T = 0.4
    sample_noise_z0_push = 0
    num_samples = 50_000

    all_images = []
    while True:
        if cond:
            labels = torch.eye(model.label_dim, device=device)[
                torch.randint(model.label_dim, size=[batch_size], device=device)]
        else:
            labels = None
        x0_sample = model.sample(batch_size, device, data_shape, sample_noise_z_T=sample_noise_z_T,
                                 sample_noise_z0_after_push=sample_noise_z0_push, labels=labels)
        images = x0_sample[0].detach().cpu()
        for img in images:
            all_images.append((img * 127.5 + 128).clip(0, 255).to(torch.uint8).float().div(255).numpy())
            i += 1
            if i >= num_samples:
                break
        if i >= num_samples:
            break

    all_images = np.stack(all_images, axis=0)

    # Compute FID & IS
    (IS, IS_std), FID = get_inception_score_and_fid(torch.tensor(all_images),
                                                    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_04_14_17_44_10/cifar10-32x32.npy')
    print(f'FID {FID:0.2f}, IS {IS:0.2f}.')
