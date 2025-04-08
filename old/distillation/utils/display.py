import numpy as np
import torch
from matplotlib import pyplot as plt


# def plot_spectrum(C, output_dir, logger):
#     if C.dtype in [torch.float32, torch.float64]:
#         C = C.detach().cpu().numpy()
#
#     D = np.linalg.eigvals(C)
#
#     plt.close('all')
#     fig = plt.figure()
#     plt.plot(np.real(D), np.imag(D), 'o', color='#444444', alpha=0.45)
#
#     ax = plt.gca()
#     from matplotlib.patches import Circle
#
#     circle = Circle((0.0, 0.0), 1.0, fill=False)
#     ax.add_artist(circle)
#
#     ax.set_aspect('equal')
#     ax.set_xticks([-1, 0, 1])
#     ax.set_yticks([-1, 0, 1])
#     plt.xlabel('Real component')
#     plt.ylabel('Imaginary component')
#
#     logger.log(f'{output_dir}/spec', fig)



import random

import torch.utils.data
import torch.nn.init
import numpy as np
from functools import wraps

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import seaborn as sns



def plotwrapper(fun):
    """Decorator that adds figure and axes handles to the kwargs of a function."""

    @wraps(fun)
    def wrapper(*args, **kwargs):

        if 'ax' not in kwargs:
            if 'fig' not in kwargs:
                figsize = kwargs['figsize'] if 'figsize' in kwargs else None
                kwargs['fig'] = plt.figure(figsize=figsize)
            kwargs['ax'] = kwargs['fig'].add_subplot(111)
        else:
            if 'fig' not in kwargs:
                kwargs['fig'] = kwargs['ax'].get_figure()

        return fun(*args, **kwargs)

    return wrapper


def axwrapper(fun):
    """Decorator that adds an axes handle to kwargs."""

    @wraps(fun)
    def wrapper(*args, **kwargs):
        if 'ax' not in kwargs:
            if 'fig' not in kwargs:
                kwargs['fig'] = plt.gcf()
            kwargs['ax'] = plt.gca()
        else:
            if 'fig' not in kwargs:
                kwargs['fig'] = kwargs['ax'].get_figure()
        return fun(*args, **kwargs)

    return wrapper


@axwrapper
def nospines(left=False, bottom=False, top=True, right=True, **kwargs):
    """
  Hides the specified axis spines (by default, right and top spines)
  """

    ax = kwargs['ax']

    # assemble args into dict
    disabled = dict(left=left, right=right, top=top, bottom=bottom)

    # disable spines
    for key in disabled:
        if disabled[key]:
            ax.spines[key].set_color('none')

    # disable xticks
    if disabled['top'] and disabled['bottom']:
        ax.set_xticks([])
    elif disabled['top']:
        ax.xaxis.set_ticks_position('bottom')
    elif disabled['bottom']:
        ax.xaxis.set_ticks_position('top')

    # disable yticks
    if disabled['left'] and disabled['right']:
        ax.set_yticks([])
    elif disabled['left']:
        ax.yaxis.set_ticks_position('right')
    elif disabled['right']:
        ax.yaxis.set_ticks_position('left')

    return ax


@axwrapper
def breathe(xlims=None, ylims=None, padding_percent=0.05, direction='out', **kwargs):
    """Adds space between axes and plot."""
    ax = kwargs['ax']

    if ax.get_xscale() == 'log':
        xfwd = np.log10
        xrev = lambda x: 10 ** x
    else:
        xfwd = lambda x: x
        xrev = lambda x: x

    if ax.get_yscale() == 'log':
        yfwd = np.log10
        yrev = lambda x: 10 ** x
    else:
        yfwd = lambda x: x
        yrev = lambda x: x

    xmin, xmax = xfwd(ax.get_xlim()) if xlims is None else xlims
    ymin, ymax = yfwd(ax.get_ylim()) if ylims is None else ylims

    xdelta = (xmax - xmin) * padding_percent
    ydelta = (ymax - ymin) * padding_percent

    ax.set_xlim(xrev(xmin - xdelta), xrev(xmax + xdelta))
    ax.spines['bottom'].set_bounds(xrev(xmin), xrev(xmax))

    ax.set_ylim(yrev(ymin - ydelta), yrev(ymax + ydelta))
    ax.spines['left'].set_bounds(yrev(ymin), yrev(ymax))

    nospines(**kwargs)

    return ax


def plot_spectrum(C, output_dir, logger):
    if C.dtype in [torch.float32, torch.float64]:
        C = C.detach().cpu().numpy()

    D = np.linalg.eigvals(C)

    plt.close('all')
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    def circle(radius=1., **kwargs):
        """Plots a unit circle."""
        ax = kwargs['ax']
        theta = np.linspace(0, 2 * np.pi, 1001)
        ax.plot(radius * np.cos(theta), radius * np.sin(theta), '-', linewidth=1.1, color='#444444', alpha=.5)

    axes[1].plot(np.real(D), np.imag(D), '.', color='#6cb2cc')
    axes[1].set_aspect('equal')
    axes[1].set_xticks([-1.5, 0, 1.5])
    axes[1].set_yticks([-1.5, 0, 1.5])
    circle(ax=axes[1])
    breathe(xlims=[-1.5, 1.5], ylims=[-1.5, 1.5])

    cmap = sns.color_palette("vlag", as_cmap=True)
    im = axes[0].imshow(C, cmap=cmap)
    # Create a divider for the first subplot
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes('right', size='5%', pad=0.1)
    # Adjust the layout of the colorbar
    cax.yaxis.tick_left()
    cax.yaxis.set_label_position('left')
    cax.tick_params(width=0)

    plt.colorbar(im, cax=cax)


    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.3)
    plt.title('Koopman Matrix and Spectrum')

    logger.log(f'{output_dir}/spec', fig)

    plt.show()
