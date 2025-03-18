""""
Script for Cifar10 data memory rearrange for faster loading

"""
import glob
import numpy as np
import tqdm



def rearrange_data(dataset_path, new_dataset_path):
    all_data_paths = glob.glob(dataset_path + '/*')
    all_data = []
    for p in tqdm.tqdm(all_data_paths):
        try:
            dynamics = np.load(p)['arr_0']
            all_data.append(np.stack([dynamics[0], dynamics[-1]]))
        except EOFError:
            pass
    all_data = np.stack(all_data)
    np.save(new_dataset_path, all_data)


dp = "/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_200k"
ndp = "/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_dataset_170k.npy"
rearrange_data(dp, ndp)
print("done")

# dp = "/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_400k"
# ndp = "/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_dataset_400k.npy"
# rearrange_data(dp, ndp)