import numpy as np
import torch
from matplotlib import pyplot as plt


def plot_spectrum(C):
    if C.dtype in [torch.float32, torch.float64]:
        C = C.detach().cpu().numpy()

    D = np.linalg.eigvals(C)

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

    plt.show()
    plt.clf()
    plt.close(fig)
