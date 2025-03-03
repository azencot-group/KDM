import torch.nn
import torch as th
from torch.functional import F

from koopman_distillation.other_methods.consistency_models.models.nn import append_dims, mean_flat


def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data ** 2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


class ConsistencyModel(torch.nn.Module):
    def __init__(self, model, target_model, teacher_model, teacher_diffusion, ema_scale_function, sigma_max, sigma_min,
                 rho, sigma_data, weight_schedule):
        super(ConsistencyModel, self).__init__()
        self.model = model
        self.target_model = target_model
        self.teacher_model = teacher_model
        self.teacher_diffusion = teacher_diffusion
        self.ema_scale_function = ema_scale_function
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_min = sigma_min
        self.sigma_data = sigma_data
        self.weight_schedule = weight_schedule
        self.loss_norm = "l2"  # todo - make it configurable

        # target model
        self.target_model.requires_grad_(False)
        self.target_model.train()

        self.teacher_model.requires_grad_(False)
        self.teacher_model.eval()

    def forward(self, x_0, x_T, cond=None, global_step=None):
        ema, num_scales = self.ema_scale_function(global_step)

        # --- setup functions --- #
        noise = th.randn_like(x_0)
        dims = x_0.ndim

        def denoise_fn(x, t):
            return self.denoise(self.model, x, t, distillation=True)[1]  # todo - add the denoise function

        @th.no_grad()
        def target_denoise_fn(x, t):
            return self.denoise(self.target_model, x, t, distillation=True)[1]

        @th.no_grad()
        def teacher_denoise_fn(x, t):
            return self.teacher_diffusion.denoise(self.teacher_model, x, t, distillation=False)[1]

        @th.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / append_dims(t, dims)
            samples = x + d * append_dims(next_t - t, dims)
            denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / append_dims(next_t, dims)
            samples = x + (d + next_d) * append_dims((next_t - t) / 2, dims)

            return samples

        # --- start the calculations --- #
        indices = th.randint(0, num_scales - 1, (x_0.shape[0],), device=x_0.device)
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t ** self.rho

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
                self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2 ** self.rho

        x_t = x_0 + noise * append_dims(t, dims)

        dropout_state = th.get_rng_state()
        distiller = denoise_fn(x_t, t)

        x_t2 = heun_solver(x_t, t, t2, x_0).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        distiller_target = distiller_target.detach()

        return {'distiller': distiller, 't': t, 'distiller_target': distiller_target, 'x_0': x_0}

    def loss(self, loss_comps):
        snrs = self.get_snr(loss_comps['t'])
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)
        if self.loss_norm == "l1":
            diffs = th.abs(loss_comps['distiller'] - loss_comps['distiller_target'])
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "l2":
            diffs = (loss_comps['distiller'] - loss_comps['distiller_target']) ** 2
            loss = mean_flat(diffs) * weights
        elif self.loss_norm == "lpips":
            if loss_comps['x_0'].shape[-1] < 256:
                distiller = F.interpolate(loss_comps['distiller'], size=224, mode="bilinear")
                distiller_target = F.interpolate(
                    loss_comps['distiller_target'], size=224, mode="bilinear"
                )

            loss = (
                    self.lpips_loss(
                        (loss_comps['distiller'] + 1) / 2.0,
                        (loss_comps['distiller_target'] + 1) / 2.0,
                    )
                    * weights
            )
        else:
            raise ValueError(f"Unknown loss norm {self.loss_norm}")

        loss = loss.mean()

        return {'loss': loss}

    def sample(self, batch_size, device, data_shape, sample_iter=1):
        x_T = torch.randn((batch_size, data_shape)).to(device)
        x_0 = None

        return x_0, x_T

    def denoise(self, model, x_t, sigmas, distillation):
        if not distillation:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
            ]
        else:
            c_skip, c_out, c_in = [
                append_dims(x, x_t.ndim)
                for x in self.get_scalings_for_boundary_condition(sigmas)
            ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, rescaled_t)  # the last item is just an placeholder
        denoised = c_out * model_output + c_skip * x_t
        return model_output, denoised

    def get_snr(self, sigmas):
        return sigmas ** -2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        c_skip = self.sigma_data ** 2 / (
                (sigma - self.sigma_min) ** 2 + self.sigma_data ** 2
        )
        c_out = (
                (sigma - self.sigma_min)
                * self.sigma_data
                / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        )
        c_in = 1 / (sigma ** 2 + self.sigma_data ** 2) ** 0.5
        return c_skip, c_out, c_in
