import numpy as np
import torch
from matplotlib import pyplot as plt

from koopman_distillation.model.modules.model_cifar10 import FastKoopmanMatrix


def plot_spectrum(C, output_dir, logger):

    if isinstance(C, FastKoopmanMatrix):
        lambda_mod = torch.exp(-torch.exp(C.nu_log))
        lambda_re = lambda_mod * torch.cos(torch.exp(C.theta_log))
        lambda_im = lambda_mod * torch.sin(torch.exp(C.theta_log))
        D = torch.complex(lambda_re, lambda_im).cpu().detach().numpy()
    else:
        C = C.weight.data.cpu().detach().numpy()
        D = np.linalg.eigvals(C)
        if C.dtype in [torch.float32, torch.float64]:
            C = C.detach().cpu().numpy()

    plt.close('all')
    fig = plt.figure()
    plt.plot(np.real(D), np.imag(D), 'o', color='#444444', alpha=0.45)

    ax = plt.gca()
    from matplotlib.patches import Circle

    circle = Circle((0.0, 0.0), 1.0, fill=False)
    ax.add_artist(circle)

    ax.set_aspect('equal')
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    plt.xlabel('Real component')
    plt.ylabel('Imaginary component')

    logger.log(f'{output_dir}/spec', fig)
