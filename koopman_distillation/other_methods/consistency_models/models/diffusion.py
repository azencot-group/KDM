import torch as th
from .nn import append_dims
import torch

class KarrasDenoiser:
    def __init__(
            self,
            sigma_data: float = 0.5,
            sigma_max=80.0,
            sigma_min=0.002,
            rho=7.0,
            weight_schedule="karras",
            distillation=False,
    ):
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.distillation = distillation
        self.rho = rho
        self.num_timesteps = 40

    def get_snr(self, sigmas):
        return sigmas ** -2

    def get_sigmas(self, sigmas):
        return sigmas

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

    def denoise(self, model, x_t, sigmas, distillation):
        import torch.distributed as dist

        # if not distillation:
        #     c_skip, c_out, c_in = [
        #         append_dims(x, x_t.ndim) for x in self.get_scalings(sigmas)
        #     ]
        # else:
        #     c_skip, c_out, c_in = [
        #         append_dims(x, x_t.ndim)
        #         for x in self.get_scalings_for_boundary_condition(sigmas)
        #     ]
        # rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        # model_output = model(c_in * x_t, rescaled_t, rescaled_t) # last item is just a placeholder
        # denoised = c_out * model_output + c_skip * x_t

        c_skip = 1
        c_out = sigmas
        c_in = 1
        c_noise = (0.5 * sigmas).log()
        F_x = model((c_in * x_t), c_noise, sigmas)
        D_x = c_skip * x_t + c_out * F_x.to(torch.float32)

        return F_x, D_x
