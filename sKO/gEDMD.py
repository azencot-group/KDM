import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.decomposition import PCA
import sympy as sp

# Define the noisy Van der Pol oscillator
def van_der_pol(t, y, mu=1.0, sigma=0.1):
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x + sigma * np.random.randn()
    return [dxdt, dvdt]

# Simulate trajectories
def simulate_vdp(n_trajectories=100, T=10, dt=0.01):
    t_eval = np.arange(0, T, dt)
    trajectories = []
    initial_conditions = []
    for _ in range(n_trajectories):
        y0 = np.random.uniform([-2, -2], [2, 2])  # Initial conditions
        sol = solve_ivp(van_der_pol, [0, T], y0, t_eval=t_eval, vectorized=True)
        trajectories.append(sol.y.T)  # Store transposed results
        initial_conditions.append(y0)  # Store transposed results
    return np.array(trajectories), t_eval, np.array(initial_conditions)


# Generate and process data
trajectories, t_eval, initial_conditions = simulate_vdp(n_trajectories=5, T=10, dt=0.05)

# Plot sample trajectories
plt.figure(figsize=(8, 6))
num_of_exp = 5
for i in range(num_of_exp):  # Plot only 20 trajectories for clarity
    plt.plot(trajectories[i, :, 0], trajectories[i, :, 1], alpha=0.5)
plt.scatter(initial_conditions[:num_of_exp, 0], initial_conditions[:num_of_exp, 1], color='red', label='Initial Conditions', alpha=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sample Trajectories of the Noisy Duffing Oscillator")
plt.legend()
plt.grid()
plt.show()


# ---- koopman ---- #

# Generate Legendre polynomial basis functions
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

