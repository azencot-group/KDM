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

from koopman_model import KoopmanModel
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


# ------- model -------
# training arguments
lr = 0.001
batch_size = 4096
iterations = 20001
print_every = 2000
hidden_dim = 64
time_steps = 2

# velocity field model init
km = MultiStepKoopmanModelDMD(hidden_dim=hidden_dim, time_steps=time_steps).to(device)


# save model with date and time
state_dict = torch.load(f'./koopman_model_20250105-175337.pt')
km.load_state_dict(state_dict)


x0_sample = km.sample(batch_size, device)
x0_sample = x0_sample[0].detach().cpu().numpy()
# plot all the points in the batch
plt.scatter(x0_sample[:, 0], x0_sample[:, 1], c='r', s=1)
plt.show()

x0 = inf_train_gen(batch_size, device)
x0 = x0.detach().cpu().numpy()
# plot all the points in the batch
plt.scatter(x0[:, 0], x0[:, 1], c='b', s=1)
plt.show()

print(f'sample loss: {((x0_sample - x0) ** 2).mean()}')
