import pickle
import os

import scipy
import torch
import tqdm

import numpy as np

from edm.dnnlib.util import open_url
from koopman_distillation.data.data_loading.data_loaders import load_data_for_testing


def translate_to_image_format(images):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return images_np


def calculate_inception_stats(image_path, num_workers=6, device=torch.device('cuda'), batch_size=32):
    # Load Inception-v3 model.
    # This is a direct PyTorch translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # dist.print0('Loading Inception-v3 model...')
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True)
    feature_dim = 2048
    with open_url(detector_url) as f:
        detector_net = pickle.load(f).to(device)

    dataset_obj = load_data_for_testing(image_path)
    data_loader = torch.utils.data.DataLoader(dataset_obj, num_workers=num_workers, batch_size=batch_size)

    # Accumulate statistics.
    print(f'Calculating statistics for {len(dataset_obj)} images...')
    mu = torch.zeros([feature_dim], dtype=torch.float64, device=device)
    sigma = torch.zeros([feature_dim, feature_dim], dtype=torch.float64, device=device)
    for images in tqdm.tqdm(data_loader):
        images = torch.tensor(translate_to_image_format(images)).permute(0, 3, 1, 2)
        if images.shape[0] == 0:
            continue
        if images.shape[1] == 1:
            images = images.repeat([1, 3, 1, 1])
        features = detector_net(images.to(device), **detector_kwargs).to(torch.float64)
        mu += features.sum(0)
        sigma += features.T @ features

    # Calculate grand totals.
    # torch.distributed.all_reduce(mu)
    # torch.distributed.all_reduce(sigma)
    mu /= len(dataset_obj)
    sigma -= mu.ger(mu) * len(dataset_obj)
    sigma /= len(dataset_obj) - 1
    return mu.cpu().numpy(), sigma.cpu().numpy()


def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))


def calculate_fid(ref_path, image_path, batch_size=32):
    with open_url(ref_path) as f:
        ref = dict(np.load(f))

    print('calculating mu and sigma...')
    mu, sigma = calculate_inception_stats(image_path=image_path, batch_size=batch_size)
    print('Calculating FID...')
    fid = calculate_fid_from_inception_stats(mu, sigma, ref['mu'], ref['sigma'])
    print(f'{fid:g}')
    return fid


def sample_and_calculate_fid(model, data_shape, num_samples, device, batch_size, epoch, image_dir, data_loader):
    i = 0
    output_dir = image_dir + '/samples'
    os.makedirs(output_dir, exist_ok=True)
    data_iter = iter(data_loader)
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            # If data_iter is exhausted, reinitialize it
            data_iter = iter(data_loader)
            batch = next(data_iter)  # Try again after reinitialization
        x0_sample = model.sample(batch_size, device, data_shape, data_batch=batch)
        images = x0_sample[0].detach().cpu().numpy()
        for img in images:
            np.savez_compressed(f'{output_dir}/img{i}', img)
            i += 1
            if i >= num_samples:
                break
        if i >= num_samples:
            break

    return calculate_fid(
        ref_path='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz',
        image_path=output_dir)
