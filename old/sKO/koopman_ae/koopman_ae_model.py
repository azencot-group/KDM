import os, sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn


def normalize_tensor(tensor):
    mean = tensor.mean(dim=0, keepdim=True)
    std = tensor.std(dim=0, keepdim=True) + 1e-4  # Avoid division by zero
    return (tensor - mean) / std


def clip_tensor(tensor, min_val=-1.0, max_val=1.0):
    return torch.clamp(tensor, min=min_val, max=max_val)


def normalize_and_clip(tensor, min_val=-6.0, max_val=6.0):
    clipped = clip_tensor(tensor, min_val, max_val)
    # normalized = normalize_tensor(clipped)
    return clipped


class KoopmanAE(nn.Module):

    def __init__(self, hidden_dim=256):
        super(KoopmanAE, self).__init__()

        self.encoder = encNet(hidden_dim=hidden_dim)
        self.drop = torch.nn.Dropout(0.1)
        self.dynamics = KoopmanLayer(hidden_dim)
        self.decoder = decNet(hidden_dim=hidden_dim)

        self.loss_func = nn.MSELoss()

        self.names = ['total', 'rec', 'predict_ambient', 'predict_latent', 'eigs']

    def forward(self, X, train=True):
        # ----- X.shape: b x t x c x w x h ------
        Z = self.encoder(X)
        Z2, Ct, masks = self.dynamics(Z)
        Z = self.drop(Z)
        X_dec = self.decoder(Z)
        X_dec2 = self.decoder(Z2)

        return X_dec, X_dec2, Z, Z2, Ct, masks

    def decode(self, Z):
        X_dec = self.decoder(Z)

        return X_dec

    def loss(self, X, outputs):
        X_dec, X_dec2, Z, Z2, Ct, masks = outputs

        # PENALTIES
        a1 = 0
        a2 = 1
        a3 = 1

        # reconstruction
        E1 = self.loss_func(X, X_dec)

        # Koopman losses
        E2, E3 = self.dynamics.loss(X, X_dec2, Z, Z2, Ct, masks)

        # LOSS
        loss = a1 * E1 + a2 * E2 + a3 * E3

        return loss, E1, E2, E3

    def predict_from_init(self, initial_states, X, cut_higher=False):
        _, _, _, _, K, _ = self.forward(X.float())
        projected_initial_states = self.encoder(initial_states.float())
        gedmd_trajectories = []
        for ic, x in zip(projected_initial_states, initial_states):
            z_traj = [ic]
            x_trag = [x.detach().cpu().numpy()]
            for _ in range(X.shape[1] - 1):
                z_next = z_traj[-1] @ K  # Ensure proper dimension match
                z_next = normalize_and_clip(z_next)
                z_traj.append(z_next)
                # if not nan
                if torch.sum(torch.isnan(z_next)) == 0:
                    if cut_higher:
                        if z_next.abs().max() < 100:
                            x_trag.append(self.decoder(z_next).detach().cpu().numpy())
                    else:
                        x_trag.append(self.decoder(z_next).detach().cpu().numpy())
            gedmd_trajectories.append(np.stack(x_trag))

        return gedmd_trajectories


class encNet(nn.Module):
    # simple MLP encoder
    def __init__(self, input_dim=2, hidden_dim=256):
        super(encNet, self).__init__()
        # encoder based only on mlps
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x):
        return self.fc(x)


class decNet(nn.Module):
    # similar ro the encNet but return into the output dim
    def __init__(self, output_dim=2, hidden_dim=256):
        super(decNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h):
        return self.fc(h)


class KoopmanLayer(nn.Module):

    def __init__(self, hidden_dim=256, args=None):
        super(KoopmanLayer, self).__init__()

        self.run = None
        self.args = args
        self.n_frames = 100
        self.k_dim = hidden_dim
        self.c_times = 100

        # loss functions
        self.loss_func = nn.MSELoss()

    def forward(self, Z):
        # Z is in b * t x c x 1 x 1
        Zr = Z.squeeze().reshape(-1, self.n_frames, self.k_dim)

        # split
        X, Y = Zr[:, :-1], Zr[:, 1:]

        # solve linear system (broadcast)
        Ct = torch.linalg.lstsq(X.reshape(-1, self.k_dim), Y.reshape(-1, self.k_dim)).solution

        # predict (broadcast)
        Zs = []
        masks = []
        Y2 = X
        for i in range(self.c_times - 1):
            Y2 = normalize_and_clip(Y2)
            Y2 = Y2 @ Ct
            if i == 0:
                Z2 = torch.cat((X[:, :i + 1], Y2[:, :]), dim=1)
                mask = torch.cat((torch.zeros_like(X[:, :i + 1]), torch.ones_like(Y2[:, :])), dim=1)
            else:
                Z2 = torch.cat((X[:, :i + 1], Y2[:, :-i]), dim=1)
                mask = torch.cat((torch.zeros_like(X[:, :i + 1]), torch.ones_like(Y2[:, :-i])), dim=1)
            if i > -1:
                Zs.append(Z2.reshape(Z.shape))
                masks.append(mask)

        assert (torch.sum(torch.isnan(Y2)) == 0)

        return torch.stack(Zs).reshape(-1, *Z.shape[1:]), Ct, torch.stack(masks).reshape(-1, *Z.shape[1:])

    def loss(self, X_dec, X_dec2, Z, Z2, Ct, masks):
        if X_dec.shape[0] < X_dec2.shape[0]:
            # predict ambient
            X = X_dec.repeat(99, 1, 1)[masks[:, :, :2] == 1]
            X_dec2 = X_dec2[masks[:, :, :2] == 1]
            E1 = self.loss_func(X, X_dec2)

            Z = Z.repeat(99, 1, 1)[masks == 1]
            Z2 = Z2[masks == 1]

            E2 = self.loss_func(Z, Z2)

        else:
            E1 = self.loss_func(X_dec, X_dec2)
            E2 = self.loss_func(Z, Z2)

        # predict latent

        return E1, E2
