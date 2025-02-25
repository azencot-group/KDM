from koopman_distillation.evaluation.fid import calculate_fid

if __name__ == '__main__':
    # output_dir = '/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond' # edm 36 NFE's
    output_dir = '/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_edm_onestep'  # edm 3 NFE's
    calculate_fid(
        ref_path='https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz',
        image_path=output_dir,
        batch_size=128)
