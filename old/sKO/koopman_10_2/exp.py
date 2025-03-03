import os.path
import sys

import numpy as np
import torch
from numpy.array_api import meshgrid
from scipy.integrate import solve_ivp
import sympy as sp
from matplotlib import pyplot as plt
from sympy import symbols


def van_der_pol(t, x, m=0.3, epsilon=0.01):
    """
    Compute the van der Pol oscillator.
    """
    x1, x2 = x
    dx1 = x2 + np.sqrt(2 * epsilon) * np.random.randn()
    dx2 = (m * (1 - x1 ** 2) * x2 - x1) + np.sqrt(2 * epsilon) * np.random.randn()
    return [dx1, dx2]


# Simulate trajectories
def simulate_vdp(n_trajectories=400, T=10, dt=0.05, m=0.3, epsilon=0.01):
    t_eval = np.arange(0, T, dt)
    trajectories = []
    initial_conditions = []
    for _ in range(n_trajectories):
        x0 = np.random.uniform([-4, -4], [4, 4])  # Initial conditions
        sol = solve_ivp(van_der_pol, [0, T], x0, t_eval=t_eval, vectorized=True)
        trajectories.append(sol.y.T)  # Store transposed results
        initial_conditions.append(x0)  # Store transposed results
    return np.array(trajectories), t_eval, np.array(initial_conditions)


def generate_legendre_basis(degree):
    """ Generate bivariate Legendre polynomial basis up to a given degree """
    x1, x2 = sp.symbols('x1 x2')
    basis = []

    for d in range(degree + 1):
        for i in range(d + 1):
            j = d - i
            p1 = sp.legendre(i, x1)
            p2 = sp.legendre(j, x2)
            basis.append(p1 * p2)

    return [sp.lambdify((x1, x2), b, 'numpy') for b in basis]


def recursive_construction(i, j, x, y):
    left = 0
    top = 0

    if i == 0 and j == 0:
        P = np.ones_like(x)
    elif i == -1 or j == -1:
        P = np.zeros_like(x)
    else:
        P = 1
        for L in range(0, i):
            right = ((2 * L + 1) / (L + 1)) * x * P - (L / (L + 1)) * left
            left = P
            P = right

        for k in range(0, j):
            low = ((2 * k + 1) / (k + 1)) * y * P - (k / (k + 1)) * top
            top = P
            P = low

    return P


def generate_legendre_basis_v2(degree):
    """ Generate bivariate Legendre polynomial basis up to a given degree """
    x1, x2 = sp.symbols('x1 x2')
    basis = []

    for d in range(degree + 1):
        for i in range(d + 1):
            func = recursive_construction(i, d - i)
            basis.append(func)

    return basis


def apply_basis_functions(x, legendre_basis):
    return np.array([f(x[0], x[1]) for f in legendre_basis])


def legendre_basis_functions_v2(X):
    samples = []
    for x in X:
        lifted = []
        for d in range(10 + 1):
            for i in range(d + 1):
                result = recursive_construction(i, d - i, x[:, 0], x[:, 1])
                lifted.append(result)
        samples.append(np.stack(lifted))
    return np.stack(samples).transpose(0, 2, 1)


# --- data --- #
if os.path.exists('data.pt') and os.path.exists('initial_conditions.pt'):
    data = torch.load('data.pt')
    initial_conditions = torch.load('initial_conditions.pt')
else:
    data, _, initial_conditions = simulate_vdp()
    torch.save(data, 'data.pt')
    torch.save(initial_conditions, 'initial_conditions.pt')

# --- lifting --- #
# max_degree = 10
# lg_basis = generate_legendre_basis(max_degree)
# lg_basis[0](data[0])
# lifted_data = apply_basis_functions(data, lg_basis)
lifted_data = legendre_basis_functions_v2(data)

# --- Koopman --- #
past = lifted_data[:, :-1, :]
future = lifted_data[:, 1:, :]
Ct = np.linalg.lstsq(past.reshape(-1, 66), future.reshape(-1, 66))[0]

eig_val, eig_vec = np.linalg.eig(Ct)
Xs, Ys = np.meshgrid(np.linspace(-4, 4, 100), np.linspace(-4, 4, 100))
lifted_grid = legendre_basis_functions_v2((np.stack([Xs.reshape(-1), Ys.reshape(-1)]).T)[:, None, :]).squeeze()
lifted_grid_eig = lifted_grid @ eig_vec.T
tmp = lifted_grid[:, 0]

# --- plots --- #

# # Plot sample trajectories
# plt.figure(figsize=(8, 6))
# num_of_exp = 100
# for i in range(num_of_exp):  # Plot only 20 trajectories for clarity
#     plt.plot(data[i, :, 0], data[i, :, 1], alpha=0.5, label='SDE' if i == 0 else "")
# plt.scatter(initial_conditions[:num_of_exp, 0], initial_conditions[:num_of_exp, 1], color='red',
#             label='Initial Conditions', alpha=0.5)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.title("Sample Trajectories of the Noisy Duffing Oscillator with gEDMD Approximation")
# plt.legend()
# plt.grid()
# plt.show()
