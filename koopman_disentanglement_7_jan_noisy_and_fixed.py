import time
import torch

from torch import nn, Tensor
import torch.nn.functional as F

# visualization
import matplotlib.pyplot as plt
from matplotlib import cm
# To avoide meshgrid warning
import warnings

from torch.nn import Sigmoid

from koopman_model import KoopmanModel, OneStepKoopmanModel
from our_utils import linear_beta_schedule

# todo - one step at a time progress + noisy process in the latent space.
# ------- init -------
warnings.filterwarnings("ignore", category=UserWarning, module='torch')

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

torch.manual_seed(42)


# ------- data -------
def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data.float()


# training arguments
lr = 0.001
batch_size = 4096
iterations = 20001
print_every = 2000
time_steps = 10


hidden_dim_list = [64, 256]
push_list = ['linear', 'non-linear']
noisy_latent_list = [0.0, 0.2]
data_list = ['fixed', 'sampled']

for hidden_dim in hidden_dim_list:
    for push in push_list:
        for noisy_latent in noisy_latent_list:
            for data in data_list:
                print('################################# new run #################################')
                print(f'hidden_dim: {hidden_dim}, push: {push}, noisy_latent: {noisy_latent}, data: {data}')

                # velocity field model init
                km = OneStepKoopmanModel(hidden_dim=hidden_dim,
                                         time_steps=time_steps,
                                         noisy_latent=noisy_latent,
                                         push=push).to(device)

                # init optimizer
                optim = torch.optim.Adam(km.parameters(), lr=lr)

                if data == 'fixed':
                    # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
                    x0 = inf_train_gen(batch_size=batch_size, device=device)  # sample data
                    xT = torch.randn_like(x0).to(device)

                # train
                start_time = time.time()
                for i in range(iterations):
                    optim.zero_grad()

                    # sample time (user's responsibility)
                    t = torch.rand(x0.shape[0]).to(device)
                    if data == 'sampled':
                        # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
                        x0 = inf_train_gen(batch_size=batch_size, device=device)  # sample data
                        xT = torch.randn_like(x0).to(device)

                    # latent pass
                    lp = km(x0=x0, xT=xT)

                    # flow matching l2 loss
                    losses = km.loss(lp)

                    # optimizer step
                    losses['loss'].backward()  # backward
                    optim.step()  # update

                    # log loss
                    if (i + 1) % print_every == 0:
                        elapsed = time.time() - start_time
                        print('| iter {:6d} | {:5.2f} ms/step | loss {:8.10f} '
                              .format(i + 1, elapsed * 1000 / print_every, losses['loss'].item()))
                        # print all other losses
                        for k, v in losses.items():
                            if k != 'loss':
                                print('| {} {:8.10f} '.format(k, v.item()), end='')
                        start_time = time.time()

                # save model with date and time
                # torch.save(km.state_dict(), f'koopman_model_{time.strftime("%Y%m%d-%H%M%S")}.pt')

                x0_sample = km.sample(batch_size, device)
                x0_sample = x0_sample[0].detach().cpu().numpy()
                # plot all the points in the batch
                plt.scatter(x0_sample[:, 0], x0_sample[:, 1], c='r', s=1)
                plt.title(f'data:{data}, push:{push}, noise: {noisy_latent}, hidden_dim: {hidden_dim}\n')
                plt.show()

                x0 = inf_train_gen(batch_size, device)
                x0 = x0.detach().cpu().numpy()
                # plot all the points in the batch
                plt.scatter(x0[:, 0], x0[:, 1], c='b', s=1)
                plt.show()

                print(f'sample loss: {((x0_sample - x0) ** 2).mean()}')
