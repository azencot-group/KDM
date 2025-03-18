import copy

import torch

from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.evaluation.fid import translate_to_image_format, sample_and_calculate_fid
from koopman_distillation.utils.names import Datasets

with torch.no_grad():
    model_path = "/home/bermann/functional_mapping/koopman_distillation/results/cifar_uncond_dmdkoopman/2025_03_14_12_03_25/model.pt"
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


    # --- new sampling --- #

    x_0 = tr_batch[0].to(device)
    x_T_data = tr_batch[1].to(device)
    T = torch.ones((x_0.shape[0],)).to(x_0.device)
    t = torch.zeros((x_0.shape[0],)).to(x_0.device)
    z_0 = model.x_0_observables_encoder(x_0, t, t)
    z_T_data = model.x_T_observables_encoder(x_T_data, T, T)
    K = torch.linalg.lstsq(z_T_data.reshape(z_0.shape[0], -1), z_0.reshape(z_0.shape[0], -1)).solution
    z_T = model.x_T_observables_encoder((torch.randn_like(x_T_data) * 80) * 0.1 + x_T_data, T, T)
    zt0_push = (z_T.reshape(z_0.shape[0], -1) @ K).reshape(z_T.shape)
    xt0_push_hat = model.x0_observables_decoder(zt0_push, t, t)

    plt.imshow(translate_to_image_format(xt0_push_hat)[i])
    plt.title("new sample")
    plt.show()
