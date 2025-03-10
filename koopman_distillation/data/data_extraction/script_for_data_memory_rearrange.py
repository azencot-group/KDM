""""
Script for Cifar10 data memory rearrange for faster loading

"""
import glob
import numpy as np
import tqdm

dataset_path = "/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond"
new_dataset_path = "/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_dataset.npy"

all_data_paths = glob.glob(dataset_path + '/*')
all_data = []
for p in tqdm.tqdm(all_data_paths):
    dynamics = np.load(p)['arr_0']
    all_data.append(np.stack([dynamics[0], dynamics[-1]]))

all_data = np.stack(all_data)
np.save(new_dataset_path, all_data)
