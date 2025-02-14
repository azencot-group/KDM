import numpy as np
import torch

from distillation.utils.data import inf_train_gen
from distillation.utils.metrics import measure_distribution_distance

fm_samplings = torch.tensor(np.load('../sol.npy'))[-1]
km_sampling = torch.tensor(torch.load('../x0_sample.pt'))
km_zT_sampling = torch.tensor(torch.load('../x0_sample_psp_reczT.pt')) # not really psp

num_experiments = 5
# Storage for distances
fm_distances = []
km_distances = []
km_zT_distances = []

# Perform the experiment multiple times
for _ in range(num_experiments):
    original_data = inf_train_gen(batch_size=50000, device='cpu')
    fm_distance = measure_distribution_distance(fm_samplings, original_data, metric='mmd')
    km_distance = measure_distribution_distance(km_sampling, original_data, metric='mmd')
    km_zT_distance = measure_distribution_distance(km_zT_sampling, original_data, metric='mmd')

    fm_distances.append(fm_distance)
    km_distances.append(km_distance)
    km_zT_distances.append(km_zT_distance)

# Compute mean and standard deviation
fm_mean = np.mean(fm_distances)
fm_std = np.std(fm_distances)
km_mean = np.mean(km_distances)
km_std = np.std(km_distances)
km_zT_mean = np.mean(km_zT_distances)
km_zT_std = np.std(km_zT_distances)

# Print results
print(f'Flow Matching distance: mean = {fm_mean:.4f}, std = {fm_std:.4f}')
print(f'Koopman distance: mean = {km_mean:.4f}, std = {km_std:.4f}')
print(f'Koopman zT distance: mean = {km_zT_mean:.4f}, std = {km_zT_std:.4f}')
