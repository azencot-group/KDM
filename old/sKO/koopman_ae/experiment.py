import os.path

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from old.sKO.koopman_ae.data import generate_series_data_vdp

from old.sKO.koopman_ae.koopman_ae_model import KoopmanAE
from old.sKO.koopman_ae.training import train

# ----------------- dataset ----------------- #

# check if the dataset is already saved
new_dataset = False
if os.path.exists('./data.npy') and os.path.exists('./initial_conditions.npy') and not new_dataset:
    initial_conditions = np.load('./initial_conditions.npy')
    data = np.load('./data.npy')

else:
    data, _, initial_conditions = generate_series_data_vdp(n_trajectories=1000)
    # save the dataset
    np.save('./data.npy', data)
    np.save('./initial_conditions.npy', initial_conditions)

# visualize data
# plt.figure(figsize=(8, 6))
# num_of_exp = 5
# for i in range(num_of_exp):  # Plot only 20 trajectories for clarity
#     plt.plot(data[i, :, 0], data[i, :, 1], alpha=0.5, label='SDE' if i == 0 else "")
# plt.scatter(initial_conditions[:num_of_exp, 0], initial_conditions[:num_of_exp, 1], color='red',
#             label='Initial Conditions')
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Sample Trajectories of the Noisy Duffing Oscillator with gEDMD Approximation")
# plt.legend()
# plt.grid()
# plt.show()

# ------------------- koopman ------------------- #
km = KoopmanAE(hidden_dim=256).cuda()

# split data into train and test
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

train_loader = DataLoader(torch.tensor(train_data), num_workers=4, batch_size=128, shuffle=True, pin_memory=True, drop_last=True)
test_loader = DataLoader(torch.tensor(test_data), num_workers=4, batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
opt = torch.optim.Adam(km.parameters(), lr=1e-4)

# ------------------- training ------------------- #
train(km, opt, train_loader, test_loader, epochs=150)
torch.save(km, 'koopman_ae.pth')

# ------------------- evaluation ------------------- #
with torch.no_grad():
    gedmd_trajectories = km.predict_from_init(torch.tensor(initial_conditions[:3]).cuda(),
                                              torch.tensor(train_data[:512]).cuda(), cut_higher=True)

    # Plot sample trajectories
    plt.figure(figsize=(8, 6))
    num_of_exp = 3
    for i in range(num_of_exp):  # Plot only 20 trajectories for clarity
        plt.plot(data[i, :, 0], data[i, :, 1], alpha=0.5, label='SDE' if i == 0 else "")
        plt.plot(gedmd_trajectories[i][:, 0], gedmd_trajectories[i][:, 1], '--', alpha=0.5,
                 label='AE Approx' if i == 0 else "")
    plt.scatter(initial_conditions[:num_of_exp, 0], initial_conditions[:num_of_exp, 1], color='red',
                label='Initial Conditions', alpha=0.5)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Sample Trajectories of the Noisy Duffing Oscillator with gEDMD Approximation")
    plt.legend()
    plt.grid()
    plt.show()
