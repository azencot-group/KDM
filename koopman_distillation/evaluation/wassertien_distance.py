import torch
from scipy.stats import wasserstein_distance


def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data.float()


def wess_distance(A, B):
    dist_x = wasserstein_distance(A[:, 0].cpu().numpy(), B[:, 0].cpu().numpy())
    dist_y = wasserstein_distance(A[:, 1].cpu().numpy(), B[:, 1].cpu().numpy())
    return (dist_x + dist_y) / 2


# # --- one time run and now no need to use --- #
# # save comparison data
# original_data = inf_train_gen(batch_size=50000, device='cpu')
# # make directory
# import os
#
# os.makedirs('/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/checkerboard/',
#             exist_ok=True)
# torch.save(original_data,
#            '/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/checkerboard/original_data.pt')
#
# # calculate reference distance
# fm_samplings = torch.tensor(np.load('../sol.npy'))[-1]
# fm_distance = wess_distance(fm_samplings, original_data)
# print(f'Flow Matching distance: {fm_distance:.6f}')


def measure_wess_distance(model, device, train_loader, num_samples=40000):
    # load num_samples from the train_loader
    final_samples = []
    for i, (x0, xT, _) in enumerate(train_loader):
        xT_sample = xT
        x0_sample = x0
        samples = model.sample(num_samples, device, data_shape=(2,), data_batch=[x0_sample, xT_sample])[0]
        final_samples.append(samples)

    samples = torch.cat(final_samples, dim=0)

    original_data = torch.load(
        '/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/checkerboard/original_data.pt')

    return wess_distance(samples[:num_samples].cpu().detach(), original_data[:num_samples])
