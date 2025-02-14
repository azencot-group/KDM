# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
import torch.nn.functional as F

from our_utils import run, linear_beta_schedule
from torch_utils import persistence


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)

        if torch.rand(1) < 0.1:
            run['loss'].append(loss.mean().item())

        return loss


@persistence.persistent_class
class FunctionalLossIterativeKoopman:
    # time dependent loss
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(1000)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # get the sigmas for the diffusion process like ddpm
        t = torch.randint(0, 1000, (images.shape[0],)).to(images.device)
        noise = torch.randn_like(images)
        alphas = self.sqrt_alphas_cumprod.to(images.device)[t][:, None, None, None]
        betas = self.sqrt_one_minus_alphas_cumprod.to(images.device)[t][:, None, None, None]
        noise_images = alphas * images + betas * noise

        # project images to latent space to get their coefficients
        l_images = net.encode_image(images, t.flatten(), labels)

        # project noisy images to latent space to get their coefficients
        l_images_noisy = net.encode_noisy_image(noise_images, t.flatten(), labels)

        # calculate the matrix that transform the function coefficients to the noisy coefficient
        coef_mat = net.get_transition_matrix(l_images_noisy, t.flatten(), labels)

        # push the noisy coefficients to the  image coefficient in the latent space
        b, c, _, _ = l_images_noisy.shape
        vec_images = l_images_noisy.reshape(b * c, -1)
        coef_mat = coef_mat.reshape(b * c, coef_mat.shape[-2], coef_mat.shape[-1])

        est_l_images = torch.bmm(vec_images.unsqueeze(1), coef_mat).squeeze().reshape(l_images_noisy.shape)

        # decode the image after the transformation
        est_images = net.decode_image(est_l_images, t.flatten(), labels)

        # calculate the loss
        loss_diffusion = ((l_images - est_l_images) ** 2)
        loss_decode = ((images - est_images) ** 2)

        loss = (loss_diffusion + loss_decode) / 2

        # print this once in a while
        if torch.rand(1) < 0.1:
            run['loss'].append(loss.mean().item())
            run['loss_diffusion'].append(loss_diffusion.mean().item())
            run['loss_decode'].append(loss_decode.mean().item())

        print(f'loss: {loss.mean().item()},'
              f' loss_diffusion: {loss_diffusion.mean().item()},'
              f' loss_decode: {loss_decode.mean().item()}')

        return loss


class FunctionalLoss:
    # Adding power matrix to the loss
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(1000)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # get the sigmas for the diffusion process like ddpm
        t = torch.randint(0, 1000, (images.shape[0],)).to(images.device)
        noise = torch.randn_like(images)
        alphas = self.sqrt_alphas_cumprod.to(images.device)[t][:, None, None, None]
        betas = self.sqrt_one_minus_alphas_cumprod.to(images.device)[t][:, None, None, None]
        noise_images = alphas * images + betas * noise

        # project images to latent space to get their coefficients
        l_images = net.encode_image(images, t.flatten(), labels)

        # project noisy images to latent space to get their coefficients
        l_images_noisy = net.encode_noisy_image(noise_images, t.flatten(), labels)

        # calculate the matrix that transform the function coefficients to the noisy coefficient
        coef_mat = net.get_transition_matrix(l_images_noisy, t.flatten(), labels)

        # push the noisy coefficients to the  image coefficient in the latent space
        b, c, _, _ = l_images_noisy.shape
        vec_images = l_images_noisy.reshape(b * c, -1)
        coef_mat = coef_mat.reshape(b * c, coef_mat.shape[-2], coef_mat.shape[-1])

        est_l_images = torch.bmm(vec_images.unsqueeze(1), coef_mat).squeeze().reshape(l_images_noisy.shape)

        # decode the image after the transformation
        est_images = net.decode_image(est_l_images, t.flatten(), labels)

        # calculate the loss
        loss_diffusion = ((l_images - est_l_images) ** 2)
        loss_decode = ((images - est_images) ** 2)

        loss = (loss_diffusion + loss_decode) / 2

        # print this once in a while
        if torch.rand(1) < 0.1:
            run['loss'].append(loss.mean().item())
            run['loss_diffusion'].append(loss_diffusion.mean().item())
            run['loss_decode'].append(loss_decode.mean().item())

        print(f'loss: {loss.mean().item()},'
              f' loss_diffusion: {loss_diffusion.mean().item()},'
              f' loss_decode: {loss_decode.mean().item()}')

        return loss


class FunctionalLossV2:
    # one step at a time loss
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        beta_schedule_fn = linear_beta_schedule
        betas = beta_schedule_fn(1000)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        # get the sigmas for the diffusion process like ddpm
        t = torch.randint(1, 1000, (images.shape[0],)).to(images.device)
        noise = torch.randn_like(images)
        alphas = self.sqrt_alphas_cumprod.to(images.device)[t][:, None, None, None]
        betas = self.sqrt_one_minus_alphas_cumprod.to(images.device)[t][:, None, None, None]
        noise_images = alphas * images + betas * noise

        # project images to latent space to get their coefficients
        l_images = net.encode_image(images, t.flatten(), labels)

        # project noisy images to latent space to get their coefficients
        l_images_noisy = net.encode_noisy_image(noise_images, t.flatten(), labels)
        l_images_noisy_minus_one = net.encode_noisy_image(noise_images, t.flatten() - 1, labels)

        # calculate the matrix that transform the function coefficients to the noisy coefficient
        coef_mat = net.get_transition_matrix(l_images_noisy, t.flatten(), labels)

        # push the noisy coefficients to the  image coefficient in the latent space
        b, c, _, _ = l_images_noisy.shape
        vec_images = l_images_noisy.reshape(b * c, -1)
        coef_mat = coef_mat.reshape(b * c, coef_mat.shape[-2], coef_mat.shape[-1])

        est_l_images_noisy_minus_one = torch.bmm(vec_images.unsqueeze(1), coef_mat).squeeze().reshape(l_images_noisy.shape)

        # decode the image after the transformation
        est_images = net.decode_image(l_images, t.flatten(), labels)

        # calculate the loss
        loss_diffusion = ((est_l_images_noisy_minus_one - l_images_noisy_minus_one) ** 2)
        loss_decode = ((images - est_images) ** 2)

        loss = (loss_diffusion + loss_decode) / 2

        # print this once in a while
        if torch.rand(1) < 0.1:
            run['loss'].append(loss.mean().item())
            run['loss_diffusion'].append(loss_diffusion.mean().item())
            run['loss_decode'].append(loss_decode.mean().item())

        print(f'loss: {loss.mean().item()},'
              f' loss_diffusion: {loss_diffusion.mean().item()},'
              f' loss_decode: {loss_decode.mean().item()}')

        return loss


class FunctionalLossV0:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()

        sigma = torch.ones_like(sigma)  # todo - delete

        # weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2 todo - return

        # y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None) todo - return
        n = torch.randn_like(images) * sigma

        # project images to latent space to get their coefficients
        l_images = net.encode_image(images, sigma.flatten(), labels)

        alpha = 0.5
        noise_images = alpha * images + (1 - alpha) * n
        # project noisy images to latent space to get their coefficients
        l_images_noisy = net.encode_noisy_image(noise_images, sigma.flatten(), labels)

        # calculate the matrix that transform the function coefficients to the noisy coefficient
        coef_mat = net.get_transition_matrix(l_images_noisy, sigma.flatten(), labels)

        # push the noisy coefficients to the  image coefficient in the latent space
        b, c, _, _ = l_images_noisy.shape
        vec_images = l_images_noisy.reshape(b * c, -1)
        coef_mat = coef_mat.reshape(b * c, coef_mat.shape[-2], coef_mat.shape[-1])

        est_l_images = torch.bmm(vec_images.unsqueeze(1), coef_mat).squeeze().reshape(l_images_noisy.shape)

        # decode the image after the transformation
        est_images = net.decode_image(est_l_images, sigma.flatten(), labels)

        # calculate the loss
        loss_diffusion = ((l_images - est_l_images) ** 2)  # todo  - return the weight *
        loss_decode = ((images - est_images) ** 2)  # todo  - return the weight *
        # todo - adding perceptual loss

        loss = (loss_diffusion + loss_decode) / 2

        # print this once in a while
        if torch.rand(1) < 0.1:
            run['loss'].append(loss.mean().item())
            run['loss_diffusion'].append(loss_diffusion.mean().item())
            run['loss_decode'].append(loss_decode.mean().item())

        print(f'loss: {loss.mean().item()},'
              f' loss_diffusion: {loss_diffusion.mean().item()},'
              f' loss_decode: {loss_decode.mean().item()}')

        return loss

# ----------------------------------------------------------------------------
