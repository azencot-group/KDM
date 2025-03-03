from torch import nn, Tensor
import torch

from old.our_utils import linear_beta_schedule


def plot_2d(samples1, samples2):
    """
    Plots two sets of 2D samples on separate subplots.

    Parameters
    ----------
    samples1 : np.ndarray
        An array of shape (N, 2) representing the first set of samples.
    samples2 : np.ndarray
        An array of shape (M, 2) representing the second set of samples.
    """
    from matplotlib import pyplot as plt

    # Create a figure and two subplots, side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the first set of samples on ax1
    ax1.scatter(samples1[:, 0], samples1[:, 1], c='r', s=10, alpha=0.7)
    ax1.set_title('x0')
    ax1.set_xlabel('X coordinate')
    ax1.set_ylabel('Y coordinate')
    ax1.axis('equal')  # Keep the aspect ratio square

    # Plot the second set of samples on ax2
    ax2.scatter(samples2[:, 0], samples2[:, 1], c='b', s=10, alpha=0.7)
    ax2.set_title('xT')
    ax2.set_xlabel('X coordinate')
    ax2.set_ylabel('Y coordinate')
    ax2.axis('equal')  # Keep the aspect ratio square

    plt.tight_layout()  # Adjust subplot spacing
    plt.show()


# ------- model -------
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
        sz = x.size()
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
        sz = x.size()
        x = x.reshape(-1, self.hidden_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output


class MatrixProducer(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_dim: int = 1):
        super(MatrixProducer, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.main = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim * 2),
            Swish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            Swish(),
            nn.Linear(hidden_dim * 4, hidden_dim * 8),
            Swish(),
            nn.Linear(hidden_dim * 8, hidden_dim * 4),
            Swish(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            Swish(),
        )

        self.final = nn.Linear(hidden_dim * 2, hidden_dim ** 2)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.reshape(-1, self.time_dim).float()
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        x = torch.cat([x, t], dim=1)
        x = self.main(x)
        x = self.final(x)
        return x.reshape(x.shape[0], self.hidden_dim, self.hidden_dim)


class BatchMatrixProducer(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_dim: int = 1, batch_size=4096):
        super(BatchMatrixProducer, self).__init__()
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.batch_size = batch_size
        self.main = nn.Sequential(
            nn.Linear(hidden_dim + time_dim, hidden_dim * 2),
            Swish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            Swish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            Swish(),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            Swish(),
            nn.Linear(hidden_dim * 2, 2),
            Swish(),
        )
        self.final = nn.Sequential(
            nn.Linear(2 * self.batch_size, 2 * self.batch_size),
            Swish(),
            nn.Linear(2 * self.batch_size, 2 * self.batch_size),
            Swish(),
            nn.Linear(2 * self.batch_size, hidden_dim ** 2))

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        t = t.reshape(-1, self.time_dim).float()
        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        x = torch.cat([x, t], dim=1)
        x = self.main(x)
        x = self.final(x.reshape(1, -1))
        return x.reshape(x.shape[0], self.hidden_dim, self.hidden_dim).squeeze()


class KoopmanModel(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000):
        super(KoopmanModel, self).__init__()
        self.encoder_x0 = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.encoder_xT = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.encoder_xt = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.decoder_xt = Decoder(output_dim=2, hidden_dim=hidden_dim)
        self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)
        # normalization layer after the matrix producer
        self.normalize = nn.LayerNorm(hidden_dim)

        self.time_steps = 1000 // time_steps
        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(1000)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()[::self.time_steps]
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()[::self.time_steps]

    def forward(self, x0: Tensor, xT: Tensor):
        # --- noising process ---- #
        # get the sigmas for the diffusion process like ddpm
        t2 = torch.randint(1, 1000 // self.time_steps, (x0.shape[0],)).to(x0.device)
        t1 = t2 - 1

        alphas2 = self.sqrt_alphas_cumprod.to(x0.device)[t2][:, None]
        betas2 = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t2][:, None]
        alphas1 = self.sqrt_alphas_cumprod.to(x0.device)[t1][:, None]
        betas1 = self.sqrt_one_minus_alphas_cumprod.to(x0.device)[t1][:, None]

        xt2 = alphas2 * x0 + betas2 * xT
        xt1 = alphas1 * xT + betas1 * x0

        # --- Koopman process ---- #
        zT = self.encoder_xT(xT, torch.ones_like(t1) * 1000 // self.time_steps)
        noise = torch.randn_like(zT) * noisy_latent
        z0 = self.encoder_x0(x0, torch.ones_like(t1)) + noise
        zt1 = self.encoder_xt(xt1, t1) + noise
        zt2 = self.encoder_xt(xt2, t2) + noise

        push_mat = self.matrix_producer(torch.cat([zt2, zT], dim=1), t2)
        zt1_push = torch.bmm(zt2.unsqueeze(1), push_mat).squeeze()
        zt1_push = self.normalize(zt1_push)

        xt1_hat = self.decoder_xt(zt1_push, t1)

        return {'x0': x0, 'z0': z0, 'xt1_hat': xt1_hat, 'zt1': zt1, 'zt2': zt2, 'zt_push': zt1_push, 'xt1': xt1}

    def loss(self, xt1, xt1_hat):
        rec_xt1 = (xt1 - xt1_hat) ** 2

        loss = rec_xt1.mean()

        return {'loss': loss, 'rec_xt1': rec_xt1.mean()}

    def sample(self, batch_size, device):
        xT = torch.randn((batch_size, 2)).to(device)


        t = (torch.ones((xT.shape[0])) * 1000 // self.time_steps).to(xT.device)
        zT = self.encoder_xT(xT, t)
        zt = zT
        for i in reversed(range(2, 1000 // self.time_steps + 1)):
            t = (torch.ones((xT.shape[0])) * i).to(xT.device)
            push_mat = self.matrix_producer(torch.cat([zt, zT], dim=1), t)
            zt = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()

        x0 = self.decoder_xt(zt, torch.ones_like(t))
        return x0, zt

    # ------- training -------


class OneStepKoopmanModel(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000, noisy_latent=0.2, push='linear', rec_xT_loss=False):
        super(OneStepKoopmanModel, self).__init__()
        self.encoder_x0 = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.encoder_xT = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.decoder_x0 = Decoder(output_dim=2, hidden_dim=hidden_dim)

        if push == 'all_linear':
            self.instead_of_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

        elif push == 'non-linear':
            self.instead_of_matrix = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)  # setup 2

        elif push == 'batch_linear':
            self.matrix_producer = BatchMatrixProducer(hidden_dim=hidden_dim)

        elif push == 'sample_linear':
            self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)

        else:
            raise ValueError(f'{push}, is unknown push')

        self.noisy_latent = noisy_latent
        self.push = push

        self.rec_zT = rec_xT_loss
        if rec_xT_loss:
            self.decoder_xT = Decoder(output_dim=2, hidden_dim=hidden_dim)

    def forward(self, x0: Tensor, xT: Tensor):
        # --- noising process ---- #
        # get the sigmas for the diffusion process like ddpm
        t = torch.ones((x0.shape[0],)).to(x0.device)

        # --- Koopman process ---- #
        zT = self.encoder_xT(xT, t)
        z0 = self.encoder_x0(x0, t)
        zT = zT + torch.randn_like(zT) * self.noisy_latent
        z0 = z0 + torch.randn_like(z0) * self.noisy_latent

        if self.push == 'all_linear':
            zt1_push = self.instead_of_matrix(zT)

        elif self.push == 'non-linear':
            zt1_push = self.instead_of_matrix(zT, t)

        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = zT @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()

        else:
            raise ValueError('unknown push')

        xt0_hat = self.decoder_x0(z0, t)
        xT_hat = None
        if self.rec_zT:
            xT_hat = self.decoder_xT(zT, t)

        xt0_push_hat = self.decoder_x0(zt1_push, t)
        with torch.no_grad():
            xT_hat_dx0 = self.decoder_x0(zT, t)

        return {'x0': x0, 'z0': z0, 'xt0_hat': xt0_hat, 'xt0_push_hat': xt0_push_hat, 'zt1_push': zt1_push,
                'xT_hat': xT_hat, 'xT_hat_dx0': xT_hat_dx0, 'xT': xT}

    def loss(self, loss_components):
        push_latent_loss = ((loss_components['x0'] - loss_components['xt0_push_hat']) ** 2).mean()
        no_push_latent_loss = ((loss_components['x0'] - loss_components['xT_hat_dx0']) ** 2).mean()

        rec_loss = ((loss_components['x0'] - loss_components['xt0_hat']) ** 2).mean()
        latent_loss = ((loss_components['z0'] - loss_components['zt1_push']) ** 2).mean()

        loss = rec_loss + push_latent_loss + latent_loss

        rec_xT_loss = torch.tensor(0)
        if self.rec_zT:
            rec_xT_loss = ((loss_components['xT'] - loss_components['xT_hat']) ** 2).mean()
            loss += rec_xT_loss

        return {'loss': loss, 'rec_loss': rec_loss, 'latent_loss': latent_loss, 'push_latent_loss': push_latent_loss,
                'no_push_latent_loss': no_push_latent_loss, 'rec_xT_loss': rec_xT_loss}

    def sample(self, batch_size, device, sample_iter=1):
        xT = torch.randn((batch_size, 2)).to(device)
        t = torch.ones((xT.shape[0],)).to(xT.device)
        zT = self.encoder_xT(xT, t)
        if self.push == 'all_linear':
            zt1_push = self.instead_of_matrix(zT)
            for i in range(1, sample_iter):
                zt1_push = self.instead_of_matrix(zt1_push)

        elif self.push == 'non-linear':
            zt1_push = self.instead_of_matrix(zT, t)
            for i in range(1, sample_iter):
                zt1_push = self.instead_of_matrix(zt1_push, t)


        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = zT @ push_mat
            for i in range(1, sample_iter):
                push_mat = self.matrix_producer(zt1_push, t)
                zt1_push = zt1_push @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()
            for i in range(1, sample_iter):
                push_mat = self.matrix_producer(zt1_push, t)
                zt1_push = torch.bmm(zt1_push.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')
        xt0_push_hat = self.decoder_x0(zt1_push, t)

        return xt0_push_hat, xT

    # ------- training -------


class MultiStepKoopmanModel(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000, noisy_latent=0.2, push='linear', num_of_steps=1):
        super(MultiStepKoopmanModel, self).__init__()
        # self.encoder_x0 = Encoder(input_dim=2, hidden_dim=hidden_dim)
        # self.encoder_xT = Encoder(input_dim=2, hidden_dim=hidden_dim)
        # self.decoder_x0 = Decoder(output_dim=2, hidden_dim=hidden_dim)

        # create an encoder for each step
        self.encoder_xt = nn.ModuleList([Encoder(input_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])
        # create same list for decoder
        self.decoder_xt = nn.ModuleList([Decoder(output_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])

        if push == 'all_linear':
            self.instead_of_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

        elif push == 'non-linear':
            self.instead_of_matrix = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)  # setup 2

        elif push == 'batch_linear':
            self.matrix_producer = BatchMatrixProducer(hidden_dim=hidden_dim)

        elif push == 'sample_linear':
            self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)

        else:
            raise ValueError(f'{push}, is unknown push')

        self.noisy_latent = noisy_latent
        self.push = push
        self.num_of_setps = num_of_steps
        self.alpha = torch.linspace(0.05, 1, self.num_of_setps + 1).flip(0)

    def forward(self, x0: Tensor, xT: Tensor):
        # --- noising process ---- #
        # get the sigmas for the diffusion process like ddpm
        tT = torch.randint(1, self.num_of_setps + 1, (1,)).to(x0.device) * torch.ones((x0.shape[0],)).to(x0.device)
        t0 = tT - 1

        # for t, extract its alpha and beta
        alphas1 = self.alpha.to(x0.device)[t0.long()][:, None]
        alphas2 = self.alpha.to(x0.device)[tT.long()][:, None]
        x0_orig = x0
        x0 = alphas1 * x0 + (1 - alphas1) * xT
        xT = alphas2 * x0 + (1 - alphas2) * xT

        # for i in reversed(range(1, 11)):
        #     print(i)
        #     tT = i * torch.ones((x0.shape[0],)).to(x0.device)
        #     t0 = tT - 1
        #
        #     # for t, extract its alpha and beta
        #     alphas1 = self.alpha.to(x0.device)[t0.long()][:, None]
        #     alphas2 = self.alpha.to(x0.device)[tT.long()][:, None]
        #     x0_orig = x0
        #     x0_tmp = alphas1 * x0 + (1 - alphas1) * xT
        #     xT_tmp = alphas2 * x0 + (1 - alphas2) * xT
        #     print(f'x0: {alphas1[0].item()}, xT: {(1 - alphas2[0].item())}')
        #
        #     plot_2d(x0_tmp.detach().cpu(), xT_tmp.detach().cpu())

        # --- Koopman process ---- #
        zT = self.encoder_xt[int(tT[0])](xT, tT)
        z0 = self.encoder_xt[int(t0[0])](x0, t0)
        z0_orig = self.encoder_xt[0](x0_orig, torch.zeros_like(t0)) + torch.randn_like(z0) * self.noisy_latent
        # z0 = self.encoder_x0(x0, t0)
        zT = zT + torch.randn_like(zT) * self.noisy_latent
        z0 = z0  # + torch.randn_like(z0) * self.noisy_latent

        if self.push == 'all_linear':
            zt1_push = self.instead_of_matrix(zT)

        elif self.push == 'non-linear':
            zt1_push = self.instead_of_matrix(zT, tT)

        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zT, tT)
            zt1_push = zT @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zT, tT)
            zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()

        else:
            raise ValueError('unknown push')

        xt0_hat = self.decoder_xt[int(t0[0])](z0, torch.zeros_like(t0))
        with torch.no_grad():
            xt0_push_hat = self.decoder_xt[int(t0[0])](zt1_push, t0)
            xT_hat = self.decoder_xt[int(t0[0])](zT, t0)

        return {'x0': x0, 'z0': z0, 'xt0_hat': xt0_hat, 'xt0_push_hat': xt0_push_hat, 'zt1_push': zt1_push,
                'xT_hat': xT_hat}

    def loss(self, loss_components):
        rec_loss = ((loss_components['x0'] - loss_components['xt0_hat']) ** 2).mean()
        push_latent_loss = ((loss_components['x0'] - loss_components['xt0_push_hat']) ** 2).mean()
        no_push_latent_loss = ((loss_components['x0'] - loss_components['xT_hat']) ** 2).mean()
        latent_loss = ((loss_components['z0'] - loss_components['zt1_push']) ** 2).mean()

        loss = rec_loss + latent_loss

        return {'loss': loss, 'rec_loss': rec_loss, 'latent_loss': latent_loss, 'push_latent_loss': push_latent_loss,
                'no_push_latent_loss': no_push_latent_loss}

    def sample(self, batch_size, device, sample_iter=1):
        xT = torch.randn((batch_size, 2)).to(device)
        t = torch.ones((xT.shape[0],)).to(xT.device) * self.num_of_setps
        zT = self.encoder_xt[self.num_of_setps](xT, t)
        if self.push == 'all_linear':
            zt1_push = self.instead_of_matrix(zT)
            for i in reversed(range(0, self.num_of_setps)):
                zt1_push = self.instead_of_matrix(zt1_push)

        elif self.push == 'non-linear':
            zt1_push = self.instead_of_matrix(zT, t)
            for i in reversed(range(0, self.num_of_setps)):
                zt1_push = self.instead_of_matrix(zt1_push, i * torch.ones_like(t))


        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = zT @ push_mat
            for i in reversed(range(0, self.num_of_setps)):
                push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                zt1_push = zt1_push @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()
            for i in reversed(range(1, self.num_of_setps)):
                push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                zt1_push = torch.bmm(zt1_push.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')
        xt0_push_hat = self.decoder_x0(zt1_push, torch.zeros_like(t))

        return xt0_push_hat, None

    def eval_koopman_operator(self, x0: Tensor, xT: Tensor):
        with torch.no_grad():
            t = torch.ones((xT.shape[0],)).to(xT.device) * self.num_of_setps
            zT = self.encoder_xt[self.num_of_setps](xT, t)
            if self.push == 'all_linear':
                zt1_push = self.instead_of_matrix(zT)
                for i in reversed(range(0, self.num_of_setps)):
                    zt1_push = self.instead_of_matrix(zt1_push)

            elif self.push == 'non-linear':
                zt1_push = self.instead_of_matrix(zT, t)
                for i in reversed(range(1, self.num_of_setps)):
                    zt1_push = self.instead_of_matrix(zt1_push, i * torch.ones_like(t))


            elif self.push == 'batch_linear':
                push_mat = self.matrix_producer(zT, t)
                zt1_push = zT @ push_mat
                for i in reversed(range(0, self.num_of_setps)):
                    push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                    zt1_push = zt1_push @ push_mat

            elif self.push == 'sample_linear':
                push_mat = self.matrix_producer(zT, t)
                zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()
                for i in reversed(range(1, self.num_of_setps)):
                    push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                    zt1_push = torch.bmm(zt1_push.unsqueeze(1), push_mat).squeeze()
            else:
                raise ValueError('unknown push')
            xt0_push_hat = self.decoder_xt[0](zt1_push, torch.zeros_like(t))
            full_push_loss = ((xt0_push_hat - x0) ** 2).mean()

            return {'full_push_loss': full_push_loss}


class MultiStepKoopmanModelWithIterative(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000, noisy_latent=0.2, push='linear', num_of_steps=1):
        super(MultiStepKoopmanModelWithIterative, self).__init__()
        self.encoder_x0 = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.encoder_xT = Encoder(input_dim=2, hidden_dim=hidden_dim)
        self.decoder_x0 = Decoder(output_dim=2, hidden_dim=hidden_dim)

        if push == 'all_linear':
            self.instead_of_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

        elif push == 'non-linear':
            self.instead_of_matrix = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)  # setup 2

        elif push == 'batch_linear':
            self.matrix_producer = BatchMatrixProducer(hidden_dim=hidden_dim)

        elif push == 'sample_linear':
            self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)

        else:
            raise ValueError(f'{push}, is unknown push')

        self.noisy_latent = noisy_latent
        self.push = push
        self.num_of_setps = num_of_steps
        self.alpha = []
        for i in range(self.num_of_setps):
            self.alpha.append(1 / (i + 1))
        self.alpha.append(0.1)
        self.alpha = torch.tensor(self.alpha)

    def forward(self, x0: Tensor, xT: Tensor):

        t = torch.ones((xT.shape[0],)).to(xT.device) * self.num_of_setps
        zT = self.encoder_xT(xT, t)
        z0 = self.encoder_x0(x0, torch.zeros_like(t))

        zT = zT + torch.randn_like(zT) * self.noisy_latent
        z0 = z0 + torch.randn_like(z0) * self.noisy_latent

        if self.push == 'all_linear':
            zt1_push = self.instead_of_matrix(zT)
            for i in reversed(range(0, self.num_of_setps)):
                zt1_push = self.instead_of_matrix(zt1_push)

        elif self.push == 'non-linear':
            zt1_push = self.instead_of_matrix(zT, t)
            for i in reversed(range(0, self.num_of_setps)):
                zt1_push = self.instead_of_matrix(zt1_push, i * torch.ones_like(t))


        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = zT @ push_mat
            for i in reversed(range(0, self.num_of_setps)):
                push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                zt1_push = zt1_push @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()
            for i in reversed(range(1, self.num_of_setps)):
                zt1_push = zt1_push + self.noisy_latent * torch.randn_like(zt1_push)
                push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                zt1_push = torch.bmm(zt1_push.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')

        xt0_hat = self.decoder_x0(z0, torch.zeros_like(t))
        with torch.no_grad():
            xt0_push_hat = self.decoder_x0(zt1_push, torch.zeros_like(t))
            xT_hat = self.decoder_x0(zT, torch.zeros_like(t))

        return {'x0': x0, 'z0': z0, 'xt0_hat': xt0_hat, 'xt0_push_hat': xt0_push_hat, 'zt1_push': zt1_push,
                'xT_hat': xT_hat}

    def loss(self, loss_components):
        rec_loss = ((loss_components['x0'] - loss_components['xt0_hat']) ** 2).mean()
        push_latent_loss = ((loss_components['x0'] - loss_components['xt0_push_hat']) ** 2).mean()
        no_push_latent_loss = ((loss_components['x0'] - loss_components['xT_hat']) ** 2).mean()
        latent_loss = ((loss_components['z0'] - loss_components['zt1_push']) ** 2).mean()

        loss = rec_loss + latent_loss

        return {'loss': loss, 'rec_loss': rec_loss, 'latent_loss': latent_loss, 'push_latent_loss': push_latent_loss,
                'no_push_latent_loss': no_push_latent_loss}

    def sample(self, batch_size, device, sample_iter=1):
        xT = torch.randn((batch_size, 2)).to(device)
        t = torch.ones((xT.shape[0],)).to(xT.device) * self.num_of_setps
        zT = self.encoder_xT(xT, t)
        if self.push == 'all_linear':
            zt1_push = self.instead_of_matrix(zT)
            for i in reversed(range(0, self.num_of_setps)):
                zt1_push = self.instead_of_matrix(zt1_push)

        elif self.push == 'non-linear':
            zt1_push = self.instead_of_matrix(zT, t)
            for i in reversed(range(0, self.num_of_setps)):
                zt1_push = self.instead_of_matrix(zt1_push, i * torch.ones_like(t))


        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = zT @ push_mat
            for i in reversed(range(0, self.num_of_setps)):
                push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                zt1_push = zt1_push @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zT, t)
            zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()
            for i in reversed(range(1, self.num_of_setps)):
                push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                zt1_push = torch.bmm(zt1_push.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')
        xt0_push_hat = self.decoder_x0(zt1_push, torch.zeros_like(t))

        return xt0_push_hat, None

    def eval_koopman_operator(self, x0: Tensor, xT: Tensor):
        with torch.no_grad():
            t = torch.ones((xT.shape[0],)).to(xT.device) * self.num_of_setps
            zT = self.encoder_xT(xT, t)
            if self.push == 'all_linear':
                zt1_push = self.instead_of_matrix(zT)
                for i in reversed(range(0, self.num_of_setps)):
                    zt1_push = self.instead_of_matrix(zt1_push)

            elif self.push == 'non-linear':
                zt1_push = self.instead_of_matrix(zT, t)
                for i in reversed(range(0, self.num_of_setps)):
                    zt1_push = self.instead_of_matrix(zt1_push, i * torch.ones_like(t))


            elif self.push == 'batch_linear':
                push_mat = self.matrix_producer(zT, t)
                zt1_push = zT @ push_mat
                for i in reversed(range(0, self.num_of_setps)):
                    push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                    zt1_push = zt1_push @ push_mat

            elif self.push == 'sample_linear':
                push_mat = self.matrix_producer(zT, t)
                zt1_push = torch.bmm(zT.unsqueeze(1), push_mat).squeeze()
                for i in reversed(range(1, self.num_of_setps)):
                    push_mat = self.matrix_producer(zt1_push, i * torch.ones_like(t))
                    zt1_push = torch.bmm(zt1_push.unsqueeze(1), push_mat).squeeze()
            else:
                raise ValueError('unknown push')
            xt0_push_hat = self.decoder_x0(zt1_push, torch.zeros_like(t))
            full_push_loss = ((xt0_push_hat - x0) ** 2).mean()

            return {'full_push_loss': full_push_loss}


class MultiStepKoopmanModelIterativeV2(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000, noisy_latent=0.2, push='linear', num_of_steps=1):
        super(MultiStepKoopmanModelIterativeV2, self).__init__()
        # create an encoder for each step
        self.encoder_xt = nn.ModuleList([Encoder(input_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])
        # create same list for decoder
        self.decoder_xt = nn.ModuleList([Decoder(output_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])

        if push == 'all_linear':
            self.instead_of_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

        elif push == 'non-linear':
            self.instead_of_matrix = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)  # setup 2

        elif push == 'batch_linear':
            self.matrix_producer = BatchMatrixProducer(hidden_dim=hidden_dim)

        elif push == 'sample_linear':
            self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)

        else:
            raise ValueError(f'{push}, is unknown push')

        self.noisy_latent = noisy_latent
        self.push = push
        self.num_of_setps = num_of_steps
        self.alpha = torch.linspace(0.05, 1, self.num_of_setps + 1).flip(0)

    def forward(self, x0: Tensor, xT: Tensor):

        xts = []
        xts_hat = []
        zts = []
        zts_minus_one = []

        zt_noisy = self.encoder_xt[self.num_of_setps](xT, self.num_of_setps * torch.ones((x0.shape[0],)).to(x0.device))
        zt_noisy = zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent

        for i in reversed(range(0, self.num_of_setps)):
            t_minus_one = i * torch.ones((x0.shape[0],)).to(x0.device)
            zt_minus_one = self.back_push(t_minus_one, zt_noisy)
            zts_minus_one.append(zt_minus_one)

            # for t, extract its alpha and beta
            alphas1 = self.alpha.to(x0.device)[t_minus_one.long()][:, None]
            xt = alphas1 * x0 + (1 - alphas1) * xT
            xts.append(xt)

            xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
            xts_hat.append(xt_hat)

            zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent

        return {'xts': xts, 'xts_hat': xts_hat, 'zts': zts, 'zts_minus_one': zts_minus_one}

    def back_push(self, t_minus_one, zt_noisy):
        if self.push == 'all_linear':
            zt_minus_one = self.instead_of_matrix(zt_noisy)

        elif self.push == 'non-linear':
            zt_minus_one = self.instead_of_matrix(zt_noisy, t_minus_one)

        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zt_noisy, t_minus_one)
            zt_minus_one = zt_noisy @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zt_noisy, t_minus_one)
            zt_minus_one = torch.bmm(zt_noisy.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')

        return zt_minus_one

    def loss(self, loss_components):
        rec_loss = ((torch.stack(loss_components['xts'])[-1] - torch.stack(loss_components['xts_hat'])[-1]) ** 2).mean()
        # latent_loss = ((torch.stack(loss_components['zts']) - torch.stack(loss_components['zts_minus_one'])) ** 2).mean()

        loss = rec_loss

        return {'loss': loss, 'rec_loss': rec_loss}

    def sample(self, batch_size, device, sample_iter=1, xT=None):
        if xT is None:
            xT = torch.randn((batch_size, 2)).to(device)
        with torch.no_grad():
            xts_hat = []

            zt_noisy = self.encoder_xt[self.num_of_setps](xT,
                                                          self.num_of_setps * torch.ones((xT.shape[0],)).to(xT.device))
            # zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent
            for i in reversed(range(0, self.num_of_setps)):
                t_minus_one = i * torch.ones((xT.shape[0],)).to(xT.device)
                zt_minus_one = self.back_push(t_minus_one, zt_noisy)
                xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
                xts_hat.append(xt_hat)

                zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent / 5

        return xts_hat.pop(), xt_hat

    def eval_koopman_operator(self, x0: Tensor, xT: Tensor):
        with torch.no_grad():
            xts = []
            xts_hat = []
            zts = []
            zts_minus_one = []

            zt_noisy = self.encoder_xt[self.num_of_setps](xT,
                                                          self.num_of_setps * torch.ones((x0.shape[0],)).to(x0.device))
            zt_noisy = zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent

            for i in reversed(range(0, self.num_of_setps)):
                t_minus_one = i * torch.ones((x0.shape[0],)).to(x0.device)
                zt_minus_one = self.back_push(t_minus_one, zt_noisy)
                zts_minus_one.append(zt_minus_one)

                # for t, extract its alpha and beta
                alphas1 = self.alpha.to(x0.device)[t_minus_one.long()][:, None]
                xt = alphas1 * x0 + (1 - alphas1) * xT
                xts.append(xt)

                xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
                xts_hat.append(xt_hat)

                zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent

            full_push_loss = ((xts_hat.pop() - x0) ** 2).mean()

            return {'full_push_loss': full_push_loss}


class MultiStepKoopmanModelDMD(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000, noisy_latent=0.2, push='linear', num_of_steps=1):
        super(MultiStepKoopmanModelDMD, self).__init__()
        # create an encoder for each step
        self.encoder_xt = nn.ModuleList([Encoder(input_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])
        # create same list for decoder
        self.decoder_xt = nn.ModuleList([Decoder(output_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])

        if push == 'all_linear':
            self.instead_of_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

        elif push == 'non-linear':
            self.instead_of_matrix = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)  # setup 2

        elif push == 'batch_linear':
            self.matrix_producer = BatchMatrixProducer(hidden_dim=hidden_dim)

        elif push == 'sample_linear':
            self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)

        else:
            raise ValueError(f'{push}, is unknown push')

        self.noisy_latent = noisy_latent
        self.push = push
        self.num_of_setps = num_of_steps
        self.alpha = torch.linspace(0.05, 1, self.num_of_setps + 1).flip(0)

    def forward(self, x0: Tensor, xT: Tensor):

        xts = []
        zts = []
        # lift variables
        for i in reversed(range(0, self.num_of_setps + 1)):
            t = torch.ones((xT.shape[0],)).to(xT.device) * i
            # for t, extract its alpha and beta
            alphas1 = self.alpha.to(x0.device)[t.long()][:, None]
            xt = alphas1 * x0 + (1 - alphas1) * xT
            xts.append(xt)
            zt = self.encoder_xt[self.num_of_setps](xT, t)
            zts.append(zt)

        xts = torch.stack(xts).permute(1, 0, 2)
        zts = torch.stack(zts).permute(1, 0, 2)

        # Koopman matrix
        # [z_1, ..., z_T]A = [z_0, ..., z_T-1]
        zts_future = zts[:, 1:, :]  # [z_1, ..., z_T]
        zts_future = zts_future + torch.randn_like(zts_future) * self.noisy_latent
        zts_past = zts[:, :-1, :]  # [z_0, ..., z_T-1]
        zts_past_pred = self.instead_of_matrix(zts_future.reshape(-1, zts_future.shape[2])).reshape(zts_future.shape)

        # reconstruction
        xts_rec = []
        xts_push_rec = []
        for i in reversed(range(0, self.num_of_setps + 1)):
            t = i * torch.ones((x0.shape[0],)).to(x0.device)
            xt_rec = self.decoder_xt[i](zts[:, i, :], t)
            xts_rec.append(xt_rec)
            if 10 > i:
                zt_push_rec = self.decoder_xt[i](zts_past_pred[:, i, :], t)
                xts_push_rec.append(zt_push_rec)

        xts_rec = torch.stack(xts_rec).permute(1, 0, 2)
        xts_push_rec = torch.stack(xts_push_rec).permute(1, 0, 2)

        return {'xts': xts, 'xts_rec': xts_rec, 'xts_push_rec': xts_push_rec, 'zts_past_pred': zts_past_pred,
                'zts_past': zts_past}

    def back_push(self, t_minus_one, zt_noisy):
        if self.push == 'all_linear':
            zt_minus_one = self.instead_of_matrix(zt_noisy)

        elif self.push == 'non-linear':
            zt_minus_one = self.instead_of_matrix(zt_noisy, t_minus_one)

        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zt_noisy, t_minus_one)
            zt_minus_one = zt_noisy @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zt_noisy, t_minus_one)
            zt_minus_one = torch.bmm(zt_noisy.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')

        return zt_minus_one

    def loss(self, loss_components):
        rec_loss = ((loss_components['xts'] - loss_components['xts_rec']) ** 2).mean()
        latent_pred_loss = ((loss_components['zts_past_pred'] - loss_components['zts_past']) ** 2).mean()
        rec_pred_loss = ((loss_components['xts'][:, :-1, :] - loss_components['xts_push_rec']) ** 2).mean()

        loss = rec_loss + latent_pred_loss + rec_pred_loss

        return {'loss': loss, 'rec_loss': rec_loss, 'latent_pred_loss': latent_pred_loss,
                'rec_pred_loss': rec_pred_loss}

    def sample(self, batch_size, device, sample_iter=1, xT=None):
        if xT is None:
            xT = torch.randn((batch_size, 2)).to(device)
        with torch.no_grad():
            xts_hat = []

            zt_noisy = self.encoder_xt[self.num_of_setps](xT,
                                                          self.num_of_setps * torch.ones((xT.shape[0],)).to(xT.device))
            # zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent
            for i in reversed(range(0, self.num_of_setps)):
                t_minus_one = i * torch.ones((xT.shape[0],)).to(xT.device)
                zt_minus_one = self.back_push(t_minus_one, zt_noisy)
                xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
                xts_hat.append(xt_hat)

                zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent / 5

        return xts_hat.pop(), xt_hat

    def eval_koopman_operator(self, x0: Tensor, xT: Tensor):
        with torch.no_grad():
            xts = []
            xts_hat = []
            zts = []
            zts_minus_one = []

            zt_noisy = self.encoder_xt[self.num_of_setps](xT,
                                                          self.num_of_setps * torch.ones((x0.shape[0],)).to(x0.device))
            zt_noisy = zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent

            for i in reversed(range(0, self.num_of_setps + 1)):
                t_minus_one = i * torch.ones((x0.shape[0],)).to(x0.device)
                zt_minus_one = self.back_push(t_minus_one, zt_noisy)
                zts_minus_one.append(zt_minus_one)

                # for t, extract its alpha and beta
                alphas1 = self.alpha.to(x0.device)[t_minus_one.long()][:, None]
                xt = alphas1 * x0 + (1 - alphas1) * xT
                xts.append(xt)

                xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
                xts_hat.append(xt_hat)

                zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent

            full_push_loss = ((xts_hat.pop() - x0) ** 2).mean()

            return {'full_push_loss': full_push_loss}


class TwoStepKoopman(nn.Module):
    def __init__(self, hidden_dim: int = 4, time_steps=1000, noisy_latent=0.2, push='linear', num_of_steps=1):
        super(TwoStepKoopman, self).__init__()
        # create an encoder for each step
        self.encoder_xt = nn.ModuleList([Encoder(input_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])
        # create same list for decoder
        self.decoder_xt = nn.ModuleList([Decoder(output_dim=2, hidden_dim=hidden_dim) for _ in range(num_of_steps + 1)])

        if push == 'all_linear':
            self.instead_of_matrix = torch.nn.Linear(hidden_dim, hidden_dim)

        elif push == 'non-linear':
            self.instead_of_matrix = Encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)  # setup 2

        elif push == 'batch_linear':
            self.matrix_producer = BatchMatrixProducer(hidden_dim=hidden_dim)

        elif push == 'sample_linear':
            self.matrix_producer = MatrixProducer(hidden_dim=hidden_dim)

        else:
            raise ValueError(f'{push}, is unknown push')

        self.noisy_latent = noisy_latent
        self.push = push
        self.num_of_setps = num_of_steps
        self.alpha = torch.linspace(0.05, 1, self.num_of_setps + 1).flip(0)

    def forward(self, x0: Tensor, xT: Tensor):

        xts = []
        zts = []
        # lift variables
        for i in reversed(range(0, self.num_of_setps + 1)):
            t = torch.ones((xT.shape[0],)).to(xT.device) * i
            # for t, extract its alpha and beta
            alphas1 = self.alpha.to(x0.device)[t.long()][:, None]
            xt = alphas1 * x0 + (1 - alphas1) * xT
            xts.append(xt)
            zt = self.encoder_xt[self.num_of_setps](xT, t)
            zts.append(zt)

        xts = torch.stack(xts).permute(1, 0, 2)
        zts = torch.stack(zts).permute(1, 0, 2)

        # Koopman matrix
        # [z_1, ..., z_T]A = [z_0, ..., z_T-1]
        zts_future = zts[:, 1:, :]  # [z_1, ..., z_T]
        zts_future = zts_future + torch.randn_like(zts_future) * self.noisy_latent
        zts_past = zts[:, :-1, :]  # [z_0, ..., z_T-1]
        zts_past_pred = self.instead_of_matrix(zts_future.reshape(-1, zts_future.shape[2])).reshape(zts_future.shape)

        # reconstruction
        xts_rec = []
        xts_push_rec = []
        for i in reversed(range(0, self.num_of_setps + 1)):
            t = i * torch.ones((x0.shape[0],)).to(x0.device)
            xt_rec = self.decoder_xt[i](zts[:, i, :], t)
            xts_rec.append(xt_rec)
            if 10 > i:
                zt_push_rec = self.decoder_xt[i](zts_past_pred[:, i, :], t)
                xts_push_rec.append(zt_push_rec)

        xts_rec = torch.stack(xts_rec).permute(1, 0, 2)
        xts_push_rec = torch.stack(xts_push_rec).permute(1, 0, 2)

        return {'xts': xts, 'xts_rec': xts_rec, 'xts_push_rec': xts_push_rec, 'zts_past_pred': zts_past_pred,
                'zts_past': zts_past}

    def back_push(self, t_minus_one, zt_noisy):
        if self.push == 'all_linear':
            zt_minus_one = self.instead_of_matrix(zt_noisy)

        elif self.push == 'non-linear':
            zt_minus_one = self.instead_of_matrix(zt_noisy, t_minus_one)

        elif self.push == 'batch_linear':
            push_mat = self.matrix_producer(zt_noisy, t_minus_one)
            zt_minus_one = zt_noisy @ push_mat

        elif self.push == 'sample_linear':
            push_mat = self.matrix_producer(zt_noisy, t_minus_one)
            zt_minus_one = torch.bmm(zt_noisy.unsqueeze(1), push_mat).squeeze()
        else:
            raise ValueError('unknown push')

        return zt_minus_one

    def loss(self, loss_components):
        rec_loss = ((loss_components['xts'] - loss_components['xts_rec']) ** 2).mean()
        latent_pred_loss = ((loss_components['zts_past_pred'] - loss_components['zts_past']) ** 2).mean()
        rec_pred_loss = ((loss_components['xts'][:, :-1, :] - loss_components['xts_push_rec']) ** 2).mean()

        loss = rec_loss + latent_pred_loss + rec_pred_loss

        return {'loss': loss, 'rec_loss': rec_loss, 'latent_pred_loss': latent_pred_loss,
                'rec_pred_loss': rec_pred_loss}

    def sample(self, batch_size, device, sample_iter=1, xT=None):
        if xT is None:
            xT = torch.randn((batch_size, 2)).to(device)
        with torch.no_grad():
            xts_hat = []

            zt_noisy = self.encoder_xt[self.num_of_setps](xT,
                                                          self.num_of_setps * torch.ones((xT.shape[0],)).to(xT.device))
            # zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent
            for i in reversed(range(0, self.num_of_setps)):
                t_minus_one = i * torch.ones((xT.shape[0],)).to(xT.device)
                zt_minus_one = self.back_push(t_minus_one, zt_noisy)
                xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
                xts_hat.append(xt_hat)

                zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent / 5

        return xts_hat.pop(), xt_hat

    def eval_koopman_operator(self, x0: Tensor, xT: Tensor):
        with torch.no_grad():
            xts = []
            xts_hat = []
            zts = []
            zts_minus_one = []

            zt_noisy = self.encoder_xt[self.num_of_setps](xT,
                                                          self.num_of_setps * torch.ones((x0.shape[0],)).to(x0.device))
            zt_noisy = zt_noisy + torch.randn_like(zt_noisy) * self.noisy_latent

            for i in reversed(range(0, self.num_of_setps + 1)):
                t_minus_one = i * torch.ones((x0.shape[0],)).to(x0.device)
                zt_minus_one = self.back_push(t_minus_one, zt_noisy)
                zts_minus_one.append(zt_minus_one)

                # for t, extract its alpha and beta
                alphas1 = self.alpha.to(x0.device)[t_minus_one.long()][:, None]
                xt = alphas1 * x0 + (1 - alphas1) * xT
                xts.append(xt)

                xt_hat = self.decoder_xt[int(t_minus_one[0])](zt_minus_one, t_minus_one)
                xts_hat.append(xt_hat)

                zt_noisy = zt_minus_one + torch.randn_like(zt_noisy) * self.noisy_latent

            full_push_loss = ((xts_hat.pop() - x0) ** 2).mean()

            return {'full_push_loss': full_push_loss}
