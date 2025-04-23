import time
import torch
import matplotlib.pyplot as plt
import warnings
from old.distillation.koopman_model import OneStepKoopmanModel
import numpy as np

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
data = np.load('/home/bermann/functional_mapping/distillation/sol.npy')

# training arguments
lr = 0.0003
batch_size = 4096
iterations = 2301
print_every = 50
time_steps = 10
hidden_dim = 256
noisy_latent = 0
rec_xT_loss = False
# push_list = ['all_linear', 'sample_linear', 'batch_linear']
push = 'all_linear'
km = OneStepKoopmanModel(hidden_dim=hidden_dim,
                         time_steps=time_steps,
                         noisy_latent=noisy_latent,
                         push=push,
                         rec_xT_loss=rec_xT_loss).to(device)

km = km.cuda()

# init optimizer
optim = torch.optim.Adam(km.parameters(), lr=lr)

# add data to dataloader
train_loader = torch.utils.data.DataLoader(torch.tensor(data.transpose(1, 0, 2)), batch_size=batch_size, shuffle=True,
                                           drop_last=True)

# train
start_time = time.time()
for i in range(iterations):
    for batch in train_loader:
        x0 = batch[:, -1].cuda()
        xT = batch[:, 0].cuda()

        optim.zero_grad()

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
torch.save(km.state_dict(), f'koopman_model_v0_no_noise.pt')

x0_sample = km.sample(50000, device)
x0_sample = x0_sample[0].detach().cpu().numpy()
# plot all the points in the batch
plt.scatter(x0_sample[:, 0], x0_sample[:, 1], c='r', s=1)
plt.title(f'data:{data}, push:{push}, noise: {noisy_latent}, hidden_dim: {hidden_dim}\n')
plt.show()

# torch.save(x0_sample, f'x0_sampled_for_tracking.pt')
# torch.save(x0_sample[1], f'x0_sample_psp_reczT.pt')
