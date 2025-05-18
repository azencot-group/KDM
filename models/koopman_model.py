"""Model architectures and preconditioning schemes used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import torch.nn.functional as F

import torch
from torch import nn
from piq import LPIPS

from models.basic_modules import SongUNet, LinearControlModule
from utils.math import psudo_hober_loss, generate_gradual_uniform_vector
from utils.names import RecLossType, CondType, EigenSpecKoopmanLossTypes


# ------------------------------------- OUR IMPLEMENTATIONS --------------------------------------- #
class FastKoopmanMatrix(nn.Module):
    def __init__(self, state_features):
        super().__init__()
        import math
        # Define size
        self.d = state_features
        scale = 1.0 / math.sqrt(2 * self.d)

        # Orthonormal real and imaginary parts
        P_re_init = self.random_orthonormal_matrix(self.d) * scale
        P_im_init = self.random_orthonormal_matrix(self.d) * scale

        # Learnable real matrices (you will keep them real-only going forward)
        self.P_re = nn.Parameter(P_re_init)
        self.P_im = nn.Parameter(P_im_init)

        # For P_inv, still use standard Gaussian (no need to orthonormalize)
        P_inv_re = torch.randn(self.d, self.d) * scale
        P_inv_im = torch.randn(self.d, self.d) * scale

        self.P_inv_re = nn.Parameter(P_inv_re)
        self.P_inv_im = nn.Parameter(P_inv_im)

        self.rmin = 0
        self.rmax = 1
        self.max_phase = 6.283

        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(
            torch.log(-0.5 * torch.log(u1 * (self.rmax + self.rmin) * (self.rmax - self.rmin) + self.rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(self.max_phase * u2))

    def forward(self, z):
        P_re_norm = self.normalize_rows(self.P_re)
        P_im_norm = self.normalize_rows(self.P_im)
        P = self.to_real_block_matrix(P_re_norm, P_im_norm)
        P_inv = self.to_real_block_matrix(self.P_inv_re, self.P_inv_im)
        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(self.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(self.theta_log))
        lambda_ = self.to_real_block_matrix(torch.diag(lambda_re), torch.diag(lambda_im))
        z = torch.cat([z, torch.zeros_like(z)], dim=1)
        z_pushed = (z @ P_inv.T) @ lambda_.T @ P.T
        z_pushed = z_pushed[:, :self.d]

        return z_pushed

    def normalize_rows(self, x, eps=1e-8):
        return x / (x.norm(dim=-1, keepdim=True) + eps)

    def normalize_columns(self, x, eps=1e-8):
        return x / (x.norm(dim=0, keepdim=True) + eps)

    def to_real_block_matrix(self, re, im):
        # Reorganize into real-valued [2d x 2d] matrix
        top = torch.cat([re, -im], dim=1)
        bottom = torch.cat([im, re], dim=1)
        return torch.cat([top, bottom], dim=0)

    def random_orthonormal_matrix(self, size):
        q, _ = torch.linalg.qr(torch.randn(size, size))
        return q

    def get_matrix_decomposition(self):
        P_re_norm = self.normalize_rows(self.P_re)
        P_im_norm = self.normalize_rows(self.P_im)
        P = self.to_real_block_matrix(P_re_norm, P_im_norm)
        P_inv = self.to_real_block_matrix(self.P_inv_re, self.P_inv_im)
        lambda_mod = torch.exp(-torch.exp(self.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(self.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(self.theta_log))
        lambda_ = self.to_real_block_matrix(torch.diag(lambda_re), torch.diag(lambda_im))

        return P_inv.T, lambda_.T, P.T


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, advers_w=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, stride=1, padding=0),
            nn.Flatten()
        )
        self.advers_w = advers_w

    def forward(self, x):
        return self.model(x)

    def loss(self, loss_comps):
        d_real = self.model(loss_comps['x0'].detach())
        d_fake = self.model(loss_comps['x0_pushed_hat'].detach())

        real_loss = F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
        fake_loss = F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
        adv_loss = real_loss * self.advers_w + fake_loss * self.advers_w
        return {'adv_loss': adv_loss, 'real_loss': real_loss, 'fake_loss': fake_loss}


class AdversarialOneStepKoopmanCifar10(torch.nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution at input/output.
                 in_channels=3,  # Number of color channels at input.
                 out_channels=3,  # Number of color channels at output.
                 label_dim=0,  # Number of class labels, 0 = unconditional.
                 augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.
                 model_channels=128,  # Base multiplier for the number of channels.
                 channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
                 channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
                 num_blocks=4,  # Number of residual blocks per resolution.
                 attn_resolutions=[16],  # List of resolutions with self-attention.
                 dropout=0.10,  # Dropout probability of intermediate activations.
                 label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.
                 embedding_type='positional',  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 encoder_type='standard',  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 decoder_type='standard',  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 resample_filter=[1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 img_channels=3,  # Number of color channels.
                 use_fp16=False,  # Execute the underlying models at FP16 precision?
                 noisy_latent=0.2,
                 rec_loss_type=RecLossType.BOTH,
                 add_sampling_noise=1,
                 psudo_huber_c=0.03,
                 initial_noise_factor=80,
                 cond_type=CondType.OnlyEncDec,
                 advers_w=1,
                 w_latent=1,
                 w_rec=1,
                 w_push=1,
                 koopman_loss_type=EigenSpecKoopmanLossTypes.NoLoss,
                 linear_proj=None,
                 ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.add_sampling_noise = add_sampling_noise
        self.initial_noise_factor = initial_noise_factor
        self.noisy_latent = noisy_latent
        self.rec_loss_type = rec_loss_type
        self.advers_w = advers_w
        self.w_latent = w_latent
        self.w_rec = w_rec
        self.w_push = w_push
        self.koopman_loss_type = koopman_loss_type

        self.lpips = LPIPS(replace_pooling=True, reduction="none")

        self.x0_observables_encoder = SongUNet(img_resolution=img_resolution,
                                               in_channels=in_channels,
                                               out_channels=out_channels,
                                               label_dim=label_dim,
                                               augment_dim=augment_dim,
                                               model_channels=model_channels,
                                               channel_mult=channel_mult,
                                               channel_mult_emb=channel_mult_emb,
                                               num_blocks=num_blocks,
                                               attn_resolutions=attn_resolutions,
                                               dropout=dropout,
                                               label_dropout=label_dropout,
                                               embedding_type=embedding_type,
                                               channel_mult_noise=channel_mult_noise,
                                               encoder_type=encoder_type,
                                               decoder_type=decoder_type,
                                               resample_filter=resample_filter)

        self.x0_observables_decoder = SongUNet(img_resolution=img_resolution,
                                               in_channels=out_channels,
                                               out_channels=in_channels,
                                               label_dim=label_dim,
                                               augment_dim=augment_dim,
                                               model_channels=model_channels,
                                               channel_mult=channel_mult,
                                               channel_mult_emb=channel_mult_emb,
                                               num_blocks=num_blocks,
                                               attn_resolutions=attn_resolutions,
                                               dropout=dropout,
                                               label_dropout=label_dropout,
                                               embedding_type=embedding_type,
                                               channel_mult_noise=channel_mult_noise,
                                               encoder_type=encoder_type,
                                               decoder_type=decoder_type,
                                               resample_filter=resample_filter)

        self.xT_observables_encoder = SongUNet(img_resolution=img_resolution,
                                               in_channels=in_channels,
                                               out_channels=out_channels,
                                               label_dim=label_dim,
                                               augment_dim=augment_dim,
                                               model_channels=model_channels,
                                               channel_mult=channel_mult,
                                               channel_mult_emb=channel_mult_emb,
                                               num_blocks=num_blocks,
                                               attn_resolutions=attn_resolutions,
                                               dropout=dropout,
                                               label_dropout=label_dropout,
                                               embedding_type=embedding_type,
                                               channel_mult_noise=channel_mult_noise,
                                               encoder_type=encoder_type,
                                               decoder_type=decoder_type,
                                               resample_filter=resample_filter)

        if linear_proj:
            in_chan = img_resolution * img_resolution * out_channels
            out_chan = linear_proj
            self.x0_observables_encoder = LinearControlModule(self.x0_observables_encoder, in_chan, out_chan,
                                                              (out_channels, img_resolution, img_resolution),
                                                              before_main_module=False)

            self.xT_observables_encoder = LinearControlModule(self.xT_observables_encoder, in_chan, out_chan,
                                                              (out_channels, img_resolution, img_resolution),
                                                              before_main_module=False)

            self.x0_observables_decoder = LinearControlModule(self.x0_observables_decoder, out_chan, in_chan,
                                                              (out_channels, img_resolution, img_resolution),
                                                              before_main_module=True)
            self.koopman_operator = torch.nn.Linear(linear_proj, linear_proj)

        else:
            self.koopman_operator = torch.nn.Linear(img_resolution * img_resolution * out_channels,
                                                    img_resolution * img_resolution * out_channels)
        self.psudo_huber_c = psudo_huber_c

        self.cond_type = cond_type
        if cond_type == CondType.KoopmanMatrixAddition:
            self.koopman_control = torch.nn.Linear(label_dim, 32 * 32 * out_channels)

        if self.koopman_loss_type == EigenSpecKoopmanLossTypes.Uniform:
            self.uniform_vec = torch.complex(torch.rand(32 * 32 * out_channels), torch.rand(32 * 32 * out_channels))

    def forward(self, x0, xT, labels=None):
        T = torch.ones((x0.shape[0],)).to(x0.device)  # no use in one step, just a placeholder
        t = torch.zeros((x0.shape[0],)).to(x0.device)  # no use in one step, just a placeholder

        z0 = self.x0_observables_encoder(x0, t, labels)
        zT = self.xT_observables_encoder(xT, T, labels)

        z0_noisy = z0 + torch.randn_like(z0) * self.noisy_latent
        zT_noisy = zT + torch.randn_like(zT) * self.noisy_latent

        z0_pushed = self.koopman_operator(zT_noisy.reshape(x0.shape[0], -1)).reshape(z0.shape)
        if self.cond_type == CondType.KoopmanMatrixAddition:
            z0_pushed += self.koopman_control(labels).reshape(z0.shape)

        x0_pushed_hat = self.x0_observables_decoder(z0_pushed, t, labels)
        x0_hat = self.x0_observables_decoder(z0_noisy, t, labels)

        return {'x0': x0, 'xT': xT, 'z0': z0, 'zT': zT, 'z0_pushed': z0_pushed, 'x0_hat': x0_hat,
                'x0_pushed_hat': x0_pushed_hat, 'koopman_op': self.koopman_operator}

    def loss(self, loss_comps, discriminator=None):

        losses = {}
        latent_loss = ((loss_comps['z0'] - loss_comps['z0_pushed']) ** 2).mean()
        losses.update({'latent_loss': latent_loss})

        # Combine
        if self.rec_loss_type == RecLossType.BOTH:
            rec_loss_l2 = ((loss_comps['x0'] - loss_comps['x0_hat']) ** 2).mean()
            rec_loss_lpips = self.lpips((loss_comps['x0'] + 1) / 2, (loss_comps['x0_hat'] + 1) / 2).mean()
            rec_loss = rec_loss_l2 + rec_loss_lpips
            push_latent_rec_loss_l2 = ((loss_comps['x0'] - loss_comps['x0_pushed_hat']) ** 2).mean()
            push_latent_rec_loss_lpips = self.lpips((loss_comps['x0'] + 1) / 2,
                                                    (loss_comps['x0_pushed_hat'] + 1) / 2).mean()
            push_rec_loss = push_latent_rec_loss_l2 + push_latent_rec_loss_lpips
            losses.update({'rec_loss': rec_loss, 'push_rec_loss': push_rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        elif self.rec_loss_type == RecLossType.LPIPS:
            push_rec_loss = self.lpips((loss_comps['x0'] + 1) / 2,
                                       (loss_comps['x0_pushed_hat'] + 1) / 2).mean()
            rec_loss = self.lpips((loss_comps['x0'] + 1) / 2,
                                  (loss_comps['x0_hat'] + 1) / 2).mean()
            losses.update({'push_rec_loss': push_rec_loss, 'rec_loss': rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        elif self.rec_loss_type == RecLossType.L2:
            rec_loss = ((loss_comps['x0'] - loss_comps['x0_hat']) ** 2).mean()
            push_rec_loss = ((loss_comps['x0'] - loss_comps['x0_pushed_hat']) ** 2).mean()
            losses.update({'push_rec_loss': push_rec_loss, 'rec_loss': rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        elif self.rec_loss_type == RecLossType.Huber:
            rec_loss = psudo_hober_loss(loss_comps['x0'], loss_comps['x0_hat'], self.psudo_huber_c).mean()
            push_rec_loss = psudo_hober_loss(loss_comps['x0'], loss_comps['x0_pushed_hat'], self.psudo_huber_c).mean()
            losses.update({'push_rec_loss': push_rec_loss, 'rec_loss': rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        else:
            loss = None
            raise ValueError(f"Invalid rec loss type: {self.rec_loss_type}")

        if discriminator:
            d_fake = discriminator(loss_comps['x0_pushed_hat'])
            adv_loss_our = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            losses.update({'adv_loss_our': adv_loss_our})
            loss += adv_loss_our * 0.01 * self.advers_w

        if self.koopman_loss_type != EigenSpecKoopmanLossTypes.NoLoss:
            eigen_koopman_loss = self.eigen_loss(self.koopman_operator.weight)
            losses.update({'eigen_koopman_loss': eigen_koopman_loss})
            loss += eigen_koopman_loss

        losses.update({'loss': loss})

        return losses

    def eigen_loss(self, koopman_matrix):

        eig_vals = torch.linalg.eigvals(koopman_matrix)
        if self.koopman_loss_type == EigenSpecKoopmanLossTypes.Uniform:
            return torch.abs(eig_vals - self.uniform_vec.to(koopman_matrix.device)).mean()
        elif self.koopman_loss_type == EigenSpecKoopmanLossTypes.OnTheCircle:
            return torch.abs(eig_vals - 1).mean()

    def sample(self, batch_size, device, data_shape, sample_noise_zT=0, sample_noise_z0_after_push=0, labels=None):

        xT = torch.randn((batch_size, *data_shape)).to(device) * self.initial_noise_factor
        T = torch.ones((xT.shape[0],)).to(xT.device)
        t = torch.zeros((xT.shape[0],)).to(xT.device)

        zT = self.xT_observables_encoder(xT, T, labels)

        if self.add_sampling_noise > 0:
            zT = zT + torch.randn_like(zT) * sample_noise_zT

        zt0_push = self.koopman_operator(zT.reshape(xT.shape[0], -1)).reshape(zT.shape)
        if self.cond_type == CondType.KoopmanMatrixAddition:
            zt0_push += self.koopman_control(labels).reshape(zt0_push.shape)

        xt0_push_hat = self.x0_observables_decoder(zt0_push + torch.randn_like(zT) * sample_noise_z0_after_push, t,
                                                   labels)

        return xt0_push_hat, xT

    def get_koopman_operator(self):
        # always a NN layer
        return self.koopman_operator.weight.data


class AdversarialOneStepKoopmanCifar10Decomposed(torch.nn.Module):
    def __init__(self,
                 img_resolution,  # Image resolution at input/output.
                 in_channels=3,  # Number of color channels at input.
                 out_channels=3,  # Number of color channels at output.
                 label_dim=0,  # Number of class labels, 0 = unconditional.
                 augment_dim=0,  # Augmentation label dimensionality, 0 = no augmentation.

                 model_channels=128,  # Base multiplier for the number of channels.
                 channel_mult=[1, 2, 2, 2],  # Per-resolution multipliers for the number of channels.
                 channel_mult_emb=4,  # Multiplier for the dimensionality of the embedding vector.
                 num_blocks=4,  # Number of residual blocks per resolution.
                 attn_resolutions=[16],  # List of resolutions with self-attention.
                 dropout=0.10,  # Dropout probability of intermediate activations.
                 label_dropout=0,  # Dropout probability of class labels for classifier-free guidance.

                 embedding_type='positional',  # Timestep embedding type: 'positional' for DDPM++, 'fourier' for NCSN++.
                 channel_mult_noise=1,  # Timestep embedding size: 1 for DDPM++, 2 for NCSN++.
                 encoder_type='standard',  # Encoder architecture: 'standard' for DDPM++, 'residual' for NCSN++.
                 decoder_type='standard',  # Decoder architecture: 'standard' for both DDPM++ and NCSN++.
                 resample_filter=[1, 1],  # Resampling filter: [1,1] for DDPM++, [1,3,3,1] for NCSN++.
                 img_channels=3,  # Number of color channels.
                 use_fp16=False,  # Execute the underlying models at FP16 precision?
                 noisy_latent=0.2,
                 rec_loss_type=RecLossType.BOTH,
                 add_sampling_noise=1,
                 psudo_huber_c=0.03,
                 initial_noise_factor=80,
                 cond_type=CondType.OnlyEncDec,
                 advers_w=1,
                 w_latent=1,
                 w_rec=1,
                 w_push=1,
                 koopman_loss_type=EigenSpecKoopmanLossTypes.NoLoss,
                 linear_proj=None,
                 ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.add_sampling_noise = add_sampling_noise
        self.initial_noise_factor = initial_noise_factor
        self.noisy_latent = noisy_latent
        self.rec_loss_type = rec_loss_type
        self.advers_w = advers_w
        self.w_latent = w_latent
        self.w_rec = w_rec
        self.w_push = w_push
        self.koopman_loss_type = koopman_loss_type

        self.lpips = LPIPS(replace_pooling=True, reduction="none")

        self.x0_observables_encoder = SongUNet(img_resolution=img_resolution,
                                               in_channels=in_channels,
                                               out_channels=out_channels,
                                               label_dim=label_dim,
                                               augment_dim=augment_dim,
                                               model_channels=model_channels,
                                               channel_mult=channel_mult,
                                               channel_mult_emb=channel_mult_emb,
                                               num_blocks=num_blocks,
                                               attn_resolutions=attn_resolutions,
                                               dropout=dropout,
                                               label_dropout=label_dropout,
                                               embedding_type=embedding_type,
                                               channel_mult_noise=channel_mult_noise,
                                               encoder_type=encoder_type,
                                               decoder_type=decoder_type,
                                               resample_filter=resample_filter)

        self.x0_observables_decoder = SongUNet(img_resolution=img_resolution,
                                               in_channels=out_channels,
                                               out_channels=in_channels,
                                               label_dim=label_dim,
                                               augment_dim=augment_dim,
                                               model_channels=model_channels,
                                               channel_mult=channel_mult,
                                               channel_mult_emb=channel_mult_emb,
                                               num_blocks=num_blocks,
                                               attn_resolutions=attn_resolutions,
                                               dropout=dropout,
                                               label_dropout=label_dropout,
                                               embedding_type=embedding_type,
                                               channel_mult_noise=channel_mult_noise,
                                               encoder_type=encoder_type,
                                               decoder_type=decoder_type,
                                               resample_filter=resample_filter)

        self.xT_observables_encoder = SongUNet(img_resolution=img_resolution,
                                               in_channels=in_channels,
                                               out_channels=out_channels,
                                               label_dim=label_dim,
                                               augment_dim=augment_dim,
                                               model_channels=model_channels,
                                               channel_mult=channel_mult,
                                               channel_mult_emb=channel_mult_emb,
                                               num_blocks=num_blocks,
                                               attn_resolutions=attn_resolutions,
                                               dropout=dropout,
                                               label_dropout=label_dropout,
                                               embedding_type=embedding_type,
                                               channel_mult_noise=channel_mult_noise,
                                               encoder_type=encoder_type,
                                               decoder_type=decoder_type,
                                               resample_filter=resample_filter)

        if linear_proj:
            in_chan = img_resolution * img_resolution * out_channels
            out_chan = linear_proj
            self.x0_observables_encoder = LinearControlModule(self.x0_observables_encoder, in_chan, out_chan,
                                                              (out_channels, img_resolution, img_resolution),
                                                              before_main_module=False)

            self.xT_observables_encoder = LinearControlModule(self.xT_observables_encoder, in_chan, out_chan,
                                                              (out_channels, img_resolution, img_resolution),
                                                              before_main_module=False)

            self.x0_observables_decoder = LinearControlModule(self.x0_observables_decoder, out_chan, in_chan,
                                                              (out_channels, img_resolution, img_resolution),
                                                              before_main_module=True)

            self.koopman_operator = FastKoopmanMatrix(linear_proj)
        else:
            self.koopman_operator = FastKoopmanMatrix(img_resolution * img_resolution * out_channels)

        self.psudo_huber_c = psudo_huber_c

        self.cond_type = cond_type
        if cond_type == CondType.KoopmanMatrixAddition:
            self.koopman_control = torch.nn.Linear(label_dim, 32 * 32 * out_channels)

        if self.koopman_loss_type == EigenSpecKoopmanLossTypes.Uniform:
            real = torch.empty(32 * 32 * out_channels).uniform_(-1, 1)
            img = torch.empty(32 * 32 * out_channels).uniform_(-1, 1)
            self.uniform_vec = torch.complex(real, img)

        if self.koopman_loss_type == EigenSpecKoopmanLossTypes.GradualUniform:
            # Example usage:
            size = 32 * 32 * out_channels  # Define out_channels before this line
            self.uniform_vec = generate_gradual_uniform_vector(size)

    def forward(self, x0, xT, labels=None):
        T = torch.ones((x0.shape[0],)).to(x0.device)  # no use in one step, just a placeholder
        t = torch.zeros((x0.shape[0],)).to(x0.device)  # no use in one step, just a placeholder

        z0 = self.x0_observables_encoder(x0, t, labels)
        zT = self.xT_observables_encoder(xT, T, labels)

        z0_noisy = z0 + torch.randn_like(z0) * self.noisy_latent
        z_T_noisy = zT + torch.randn_like(zT) * self.noisy_latent

        z0_pushed = self.koopman_operator(z_T_noisy.reshape(x0.shape[0], -1)).reshape(z0.shape)
        if self.cond_type == CondType.KoopmanMatrixAddition:
            z0_pushed += self.koopman_control(labels).reshape(z0.shape)

        x0_pushed_hat = self.x0_observables_decoder(z0_pushed, t, labels)
        x0_hat = self.x0_observables_decoder(z0_noisy, t, labels)

        return {'x0': x0, 'xT': xT, 'z0': z0, 'z_T': zT, 'z0_pushed': z0_pushed, 'x0_hat': x0_hat,
                'x0_pushed_hat': x0_pushed_hat, 'koopman_op': self.koopman_operator}

    def loss(self, loss_comps, discriminator=None):

        losses = {}
        latent_loss = ((loss_comps['z0'] - loss_comps['z0_pushed']) ** 2).mean()
        losses.update({'latent_loss': latent_loss})

        # Combine
        if self.rec_loss_type == RecLossType.BOTH:
            rec_loss_l2 = ((loss_comps['x0'] - loss_comps['x0_hat']) ** 2).mean()
            rec_loss_lpips = self.lpips((loss_comps['x0'] + 1) / 2, (loss_comps['x0_hat'] + 1) / 2).mean()
            rec_loss = rec_loss_l2 + rec_loss_lpips
            push_latent_rec_loss_l2 = ((loss_comps['x0'] - loss_comps['x0_pushed_hat']) ** 2).mean()
            push_latent_rec_loss_lpips = self.lpips((loss_comps['x0'] + 1) / 2,
                                                    (loss_comps['x0_pushed_hat'] + 1) / 2).mean()
            push_rec_loss = push_latent_rec_loss_l2 + push_latent_rec_loss_lpips
            losses.update({'rec_loss': rec_loss, 'push_rec_loss': push_rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        elif self.rec_loss_type == RecLossType.LPIPS:
            push_rec_loss = self.lpips((loss_comps['x0'] + 1) / 2,
                                       (loss_comps['x0_pushed_hat'] + 1) / 2).mean()
            rec_loss = self.lpips((loss_comps['x0'] + 1) / 2,
                                  (loss_comps['x0_hat'] + 1) / 2).mean()
            losses.update({'push_rec_loss': push_rec_loss, 'rec_loss': rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        elif self.rec_loss_type == RecLossType.L2:
            rec_loss = ((loss_comps['x0'] - loss_comps['x0_hat']) ** 2).mean()
            push_rec_loss = ((loss_comps['x0'] - loss_comps['x0_pushed_hat']) ** 2).mean()
            losses.update({'push_rec_loss': push_rec_loss, 'rec_loss': rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        elif self.rec_loss_type == RecLossType.Huber:
            rec_loss = psudo_hober_loss(loss_comps['x0'], loss_comps['x0_hat'], self.psudo_huber_c).mean()
            push_rec_loss = psudo_hober_loss(loss_comps['x0'], loss_comps['x0_pushed_hat'], self.psudo_huber_c).mean()
            losses.update({'push_rec_loss': push_rec_loss, 'rec_loss': rec_loss})

            loss = latent_loss * self.w_latent + rec_loss * self.w_rec + push_rec_loss * self.w_push

        else:
            loss = None
            raise ValueError(f"Invalid rec loss type: {self.rec_loss_type}")

        if discriminator:
            d_fake = discriminator(loss_comps['x0_pushed_hat'])
            adv_loss_our = F.binary_cross_entropy_with_logits(d_fake, torch.ones_like(d_fake))
            losses.update({'adv_loss_our': adv_loss_our})
            loss += adv_loss_our * 0.01 * self.advers_w

        if self.koopman_loss_type != EigenSpecKoopmanLossTypes.NoLoss:
            eigen_koopman_loss = self.eigen_loss()
            losses.update({'eigen_koopman_loss': eigen_koopman_loss})
            loss += eigen_koopman_loss

        losses.update({'loss': loss})

        return losses

    def eigen_loss(self):
        # inverse loss
        P_re_norm = self.koopman_operator.normalize_rows(self.koopman_operator.P_re)
        P_im_norm = self.koopman_operator.normalize_rows(self.koopman_operator.P_im)
        P = self.koopman_operator.to_real_block_matrix(P_re_norm, P_im_norm)
        P_inv = self.koopman_operator.to_real_block_matrix(self.koopman_operator.P_inv_re,
                                                           self.koopman_operator.P_inv_im)
        d = P_re_norm.shape[0]
        eye_real = torch.eye(2 * d, device=P.device)
        inv_loss = F.mse_loss(P_inv @ P, eye_real) + \
                   F.mse_loss(P @ P_inv, eye_real)

        # eigen values extraction
        lambda_mod = torch.exp(-torch.exp(self.koopman_operator.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(self.koopman_operator.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(self.koopman_operator.theta_log))
        eig_vals = torch.complex(lambda_re, lambda_im)

        # eigen values
        if self.koopman_loss_type in [EigenSpecKoopmanLossTypes.Uniform, EigenSpecKoopmanLossTypes.GradualUniform]:
            return torch.abs(eig_vals - self.uniform_vec.to(eig_vals.device)).mean() + inv_loss

        elif self.koopman_loss_type == EigenSpecKoopmanLossTypes.OnTheCircle:
            return torch.abs(eig_vals - 1).mean() + inv_loss

    def sample(self, batch_size, device, data_shape, sample_noise_zT=0, sample_noise_z0_after_push=0, labels=None):

        xT = torch.randn((batch_size, *data_shape)).to(device) * self.initial_noise_factor

        T = torch.ones((xT.shape[0],)).to(xT.device)
        t = torch.zeros((xT.shape[0],)).to(xT.device)

        zT = self.xT_observables_encoder(xT, T, labels)

        if self.add_sampling_noise > 0:
            zT = zT + torch.randn_like(zT) * sample_noise_zT

        zt0_push = self.koopman_operator(zT.reshape(xT.shape[0], -1)).reshape(zT.shape)
        if self.cond_type == CondType.KoopmanMatrixAddition:
            zt0_push += self.koopman_control(labels).reshape(zt0_push.shape)

        xt0_push_hat = self.x0_observables_decoder(zt0_push + torch.randn_like(zT) * sample_noise_z0_after_push, t,
                                                   labels)

        return xt0_push_hat, xT

    def get_koopman_operator(self):
        # always a decomposition of 3 matrices
        P_inv, lambda_, P = self.koopman_operator.get_matrix_decomposition()
        return [P_inv.data, lambda_.data, P.data]
