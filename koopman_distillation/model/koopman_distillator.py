import torch.nn


class KoopmanDistillOneStep(torch.nn.Module):
    def __init__(self, x0_observables_encoder, x_T_observables_encoder, x0_observables_decoder, koopman_operator,
                 noisy_latent=0.2):
        super(KoopmanDistillOneStep, self).__init__()
        self.x_0_observables_encoder = x0_observables_encoder
        self.x_T_observables_encoder = x_T_observables_encoder
        self.x0_observables_decoder = x0_observables_decoder
        self.koopman_operator = koopman_operator

        self.noisy_latent = noisy_latent

    def forward(self, x_0, x_T, cond=None):
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

    def loss(self, loss_comps):
        with torch.no_grad():
            no_push_latent_rec_loss = ((loss_comps['x_0'] - loss_comps['x_T_pushed_hat']) ** 2).mean()
            rec_loss_x_T = ((loss_comps['x_T'] - loss_comps['x_T_hat']) ** 2).mean()
            push_latent_rec_loss = ((loss_comps['x_0'] - loss_comps['x_0_pushed_hat']) ** 2).mean()

        rec_loss_x_0 = ((loss_comps['x_0'] - loss_comps['x_0_hat']) ** 2).mean()
        latent_loss = ((loss_comps['z_0'] - loss_comps['z_0_pushed']) ** 2).mean()

        loss = rec_loss_x_0 + latent_loss

        return {'loss': loss, 'rec_loss_x_0': rec_loss_x_0, 'rec_loss_x_T': rec_loss_x_T, 'latent_loss': latent_loss,
                'push_latent_rec_loss': push_latent_rec_loss, 'no_push_latent_rec_loss': no_push_latent_rec_loss}

    def sample(self, batch_size, device, data_shape, sample_iter=1):
        x_T = torch.randn((batch_size, data_shape)).to(device)  # todo - generalize to other datasets
        T = torch.ones((x_T.shape[0],)).to(x_T.device)
        t = torch.zeros((x_T.shape[0],)).to(x_T.device)

        zT = self.x_T_observables_encoder(x_T, T)
        zt0_push = self.koopman_operator(zT)
        xt0_push_hat = self.x0_observables_decoder(zt0_push, t)

        return xt0_push_hat, x_T
