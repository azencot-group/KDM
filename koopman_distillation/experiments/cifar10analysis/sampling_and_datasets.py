import copy
import pickle

import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from edm.dnnlib.util import open_url
from koopman_distillation.data.data_loading.datasets_objects import Cifar10Dataset
from koopman_distillation.evaluation.fid import translate_to_image_format, calculate_fid_from_inception_stats, \
    sample_and_calculate_fid, sample_and_calculate_fid_for_test

class Cifar10DatasetTmp(torch.utils.data.Dataset):
    def __init__(self, path):
        # parse all the paths in path
        self.paths = path

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, ix):
        img = self.paths[ix]

        return [img]

def calculate_inception_stats_from_dataset_loader(dataset_loader, feature_dim=2048, device='cuda'):
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    with open_url(detector_url) as f:
        detector_net = pickle.load(f).to(device)

    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images in tqdm(dataset_loader):
        images = images[0]
        images = torch.tensor(translate_to_image_format(images)).permute(0, 3, 1, 2)
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    mu /= len(dataset_loader.dataset)
    sigma -= mu.ger(mu) * len(dataset_loader.dataset)
    sigma /= len(dataset_loader.dataset) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()



bs = 64
dataset = Cifar10Dataset('/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_1M')
test_dataset = Cifar10Dataset('/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond')
dataset1 = copy.deepcopy(dataset)
dataset2 = copy.deepcopy(dataset)

dataset1.paths = dataset1.paths[:len(dataset.paths) // 2]
dataset2.paths = dataset2.paths[len(dataset.paths) // 2:]

# randomly sample from each dataset 50k samples
total_indices = torch.randperm(100000)
indices = total_indices.tolist()
dataset1_new_paths = [dataset1.paths[i] for i in indices[:50000]]
dataset2_new_paths = [dataset2.paths[i] for i in indices[50000:]]
dataset1.paths = dataset1_new_paths
dataset2.paths = dataset2_new_paths

# dataloader for each dataset
dataloader1 = torch.utils.data.DataLoader(dataset1, batch_size=bs, num_workers=6)
dataloader2 = torch.utils.data.DataLoader(dataset2, batch_size=bs, num_workers=6)
test_dataloader = torch.utils.data.DataLoader(dataset2, batch_size=bs, num_workers=6)

with open_url('https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz') as f:
    ref = dict(np.load(f))

model = torch.load(
    '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_03_31_14_10_14/model.pt')
    # '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_03_30_08_46_49/model.pt')
model.eval()
out_dir = '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_03_31_14_10_14/tmp'
# out_dir = '/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond/2025_03_30_08_46_49/tmp'


# ---- Calculate the test distances ---- #

# real_data = []
# rec_data = []
# for batch in test_dataloader:
#     xt, xT, _ = batch
#     real_data.append(xt.detach().cpu())
#     xT = xT.cuda()
#     xt = xt.cuda()
#     T = torch.ones((xT.shape[0],)).to(xT.device)
#     t = torch.zeros((xT.shape[0],)).to(xT.device)
#     zT = model.x_T_observables_encoder(xT, T, T)
#     zt0_push = model.koopman_operator(zT.reshape(zT.shape[0], -1)).reshape(zT.shape)
#     xt0_push_hat = model.x0_observables_decoder(zt0_push, t, t)
#     rec_data.append(xt0_push_hat.detach().cpu())
#
# # stack
# real_dataset = Cifar10DatasetTmp(torch.cat(real_data, dim=0))
# rec_dataset = Cifar10DatasetTmp(torch.cat(rec_data, dim=0))
# # dataloader
# real_data = torch.utils.data.DataLoader(real_dataset, batch_size=bs, num_workers=6)
# rec_data = torch.utils.data.DataLoader(rec_dataset, batch_size=bs, num_workers=6)
# # calculate fid between them and them between the real data
# mu1, sigma1 = calculate_inception_stats_from_dataset_loader(real_data)
# mu2, sigma2 = calculate_inception_stats_from_dataset_loader(rec_data)
#
# fid = calculate_fid_from_inception_stats(mu1, sigma1, mu2, sigma2)
# print(f'1_2: {fid:g}')
# fid = calculate_fid_from_inception_stats(mu1, sigma1, ref['mu'], ref['sigma'])
# print(f'1_ref: {fid:g}')
# fid = calculate_fid_from_inception_stats(ref['mu'], ref['sigma'], mu2, sigma2)
# print(f'2_ref: {fid:g}')


 # ---- calculate FID for different noise levels ---- #
spec_j = [0, 0.001, 0.01]
spec_i = [0, 0.2, 0.4, 0.6]
for j in spec_j:
    for i in spec_i:
        fid = sample_and_calculate_fid_for_test(model=model,
                                                data_shape=(3, 32, 32),
                                                num_samples=50000,
                                                device='cuda',
                                                batch_size=bs,
                                                epoch=0,
                                                image_dir=out_dir,
                                                data_loader=None,
                                                sample_noise_z_T=i,
                                                sample_noise_z0_push=j)

        print(f'i: {i}, j: {j}, fid: {fid}')




mu1, sigma1 = calculate_inception_stats_from_dataset_loader(dataloader1)
mu2, sigma2 = calculate_inception_stats_from_dataset_loader(dataloader2)
fid = calculate_fid_from_inception_stats(mu1, sigma1, mu2, sigma2)
print(f'1_2: {fid:g}')
fid = calculate_fid_from_inception_stats(mu1, sigma1, ref['mu'], ref['sigma'])
print(f'1_ref: {fid:g}')
fid = calculate_fid_from_inception_stats(ref['mu'], ref['sigma'], mu2, sigma2)
print(f'2_ref: {fid:g}')
