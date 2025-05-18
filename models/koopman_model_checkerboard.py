from torch import nn, Tensor
import torch


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


class Encoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 4, time_dim: int = 1):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output


class Decoder(nn.Module):
    def __init__(self, output_dim: int = 2, hidden_dim: int = 4, time_dim: int = 1):
        super().__init__()

        self.output_dim = output_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        x = x.reshape(-1, self.hidden_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output


class KoopmanDistillModelCheckerBoard(torch.nn.Module):
    def __init__(self, x0_observables_encoder, x_T_observables_encoder, x0_observables_decoder, koopman_operator,
                 rec_loss_type, noisy_latent=0.4):
        super(KoopmanDistillModelCheckerBoard, self).__init__()
        self.x_0_observables_encoder = x0_observables_encoder
        self.x_T_observables_encoder = x_T_observables_encoder
        self.x0_observables_decoder = x0_observables_decoder
        self.koopman_operator = koopman_operator

        self.noisy_latent = noisy_latent
        self.rec_loss_type = rec_loss_type

    def forward(self, x_0, x_T, labels=None):
        T = torch.ones((x_0.shape[0],)).to(x_0.device)  # no use in one step, just a placeholder
        t = torch.zeros((x_0.shape[0],)).to(x_0.device)  # no use in one step, just a placeholder

        # the dynamical system start at state S in time 'T' in go backward in time to time 't'
        z_0 = self.x_0_observables_encoder(x_0, t)
        z_T = self.x_T_observables_encoder(x_T, T)

        z_0_noisy = z_0 + torch.randn_like(z_0) * self.noisy_latent
        z_T_noisy = z_T + torch.randn_like(z_T) * self.noisy_latent

        z_0_pushed = self.koopman_operator(z_T_noisy)

        with torch.no_grad():
            x_0_pushed_hat = self.x0_observables_decoder(z_0_noisy, t)
            x_T_pushed_hat = self.x0_observables_decoder(z_T, t)
            x_T_hat = self.x0_observables_decoder(z_T_noisy, T)

        x_0_hat = self.x0_observables_decoder(z_0_noisy, t)

        return {'x_0': x_0, 'x_T': x_T, 'z_0': z_0, 'z_T': z_T, 'z_0_pushed': z_0_pushed, 'x_0_hat': x_0_hat,
                'x_0_pushed_hat': x_0_pushed_hat, 'x_T_pushed_hat': x_T_pushed_hat, 'x_T_hat': x_T_hat,
                'koopman_op': self.koopman_operator}

    def loss(self, loss_comps, discriminator=None):
        with torch.no_grad():
            no_push_latent_rec_loss = ((loss_comps['x_0'] - loss_comps['x_T_pushed_hat']) ** 2).mean()
            rec_loss_x_T = ((loss_comps['x_T'] - loss_comps['x_T_hat']) ** 2).mean()
            push_latent_rec_loss = ((loss_comps['x_0'] - loss_comps['x_0_pushed_hat']) ** 2).mean()

        rec_loss_x_0 = ((loss_comps['x_0'] - loss_comps['x_0_hat']) ** 2).mean()
        latent_loss = ((loss_comps['z_0'] - loss_comps['z_0_pushed']) ** 2).mean()

        loss = rec_loss_x_0 + latent_loss

        return {'loss': loss, 'rec_loss_x_0': rec_loss_x_0, 'rec_loss_x_T': rec_loss_x_T, 'latent_loss': latent_loss,
                'push_latent_rec_loss': push_latent_rec_loss, 'no_push_latent_rec_loss': no_push_latent_rec_loss}

    def sample(self, batch_size, device, data_shape=(2,)):
        x_T = torch.randn((batch_size, *data_shape)).to(device)
        T = torch.ones((x_T.shape[0],)).to(x_T.device)
        t = torch.zeros((x_T.shape[0],)).to(x_T.device)

        zT = self.x_T_observables_encoder(x_T, T)
        zt0_push = self.koopman_operator(zT)
        xt0_push_hat = self.x0_observables_decoder(zt0_push, t)

        return xt0_push_hat, x_T
