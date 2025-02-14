import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.decomposition import PCA
import itertools
import sympy as sp

# Define parameters
alpha = 1
beta = -1
delta = 0.5
epsilon = 0.0025
T = 10  # Simulation time
N_trajectories = 400  # Number of independent trajectories


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


# Define basis functions
max_degree = 10
legendre_basis = generate_legendre_basis(max_degree)



def basis_functions(x):
    return np.array([f(x[0], x[1]) for f in legendre_basis])


def duffing_sde(t, state, noise, mu=1.0, sigma=0.1):
    """ Stochastic Duffing oscillator system """
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x + sigma * np.random.randn()
    return np.array([dxdt, dvdt])



# Generate trajectories
t_eval = np.linspace(0, T, 100)  # Time points
initial_conditions = np.random.uniform(-2.5, 2.5, (N_trajectories, 2))
trajectories = []
koopman_data_X = []
koopman_data_Y = []

for ic in initial_conditions:
    x1_traj, x2_traj = [ic[0]], [ic[1]]
    dt = t_eval[1] - t_eval[0]

    for t in t_eval[:-1]:
        noise = np.random.normal(0, np.sqrt(dt))  # Brownian motion
        state = np.array([x1_traj[-1], x2_traj[-1]])
        dx = duffing_sde(t, state, noise) * dt
        x1_traj.append(state[0] + dx[0])
        x2_traj.append(state[1] + dx[1])

        koopman_data_X.append(basis_functions(state))
        koopman_data_Y.append(basis_functions(state + dx))

    trajectories.append(np.array([x1_traj, x2_traj]))

trajectories = np.array(trajectories)

# Koopman Operator Approximation using gEDMD
koopman_data_X = np.array(koopman_data_X)
koopman_data_Y = np.array(koopman_data_Y)
koopman_model = LinearRegression(fit_intercept=False)
koopman_model.fit(koopman_data_X, koopman_data_Y)
K_matrix = koopman_model.coef_.T  # Ensure square Koopman matrix

# Eigenfunction decomposition using eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(K_matrix)


# Generate inverse Legendre polynomial basis functions
def generate_inverse_legendre_basis():
    """ Learn a regression model to invert the Legendre transformation back to (x1, x2). """
    X_train = np.array([basis_functions([xi, yi]) for xi, yi in initial_conditions])
    Y_train = np.array(initial_conditions)  # (x1, x2) original states

    inverse_model = Ridge(alpha=1e-6)  # Ridge regression for stability
    inverse_model.fit(X_train, Y_train)
    return inverse_model


# Initialize inverse transformation model
inverse_model = generate_inverse_legendre_basis()

# Generate approximated Koopman trajectories using eigenfunctions
gedmd_trajectories = []
for ic in initial_conditions[:200]:
    x_traj = [basis_functions(ic)]
    for _ in range(len(t_eval) - 1):
        x_next = K_matrix @ x_traj[-1]  # Ensure proper dimension match
        x_traj.append(x_next)

    res = inverse_model.predict(np.array(x_traj)).T
    gedmd_trajectories.append(res)

# sort the list by the maximum number, put the lowest number first
gedmd_trajectories = sorted(gedmd_trajectories, key=lambda x: np.max(x))
gedmd_trajectories = np.array(gedmd_trajectories)

# Plot sample trajectories
plt.figure(figsize=(8, 6))
num_of_exp = 5
for i in range(num_of_exp):  # Plot only 20 trajectories for clarity
    plt.plot(trajectories[i, 0, :], trajectories[i, 1, :], alpha=0.5, label='SDE' if i == 0 else "")
    plt.plot(gedmd_trajectories[i, 0, :], gedmd_trajectories[i, 1, :], '--', alpha=0.5,
             label='gEDMD Approx' if i == 0 else "")
plt.scatter(initial_conditions[:num_of_exp, 0], initial_conditions[:num_of_exp, 1], color='red', label='Initial Conditions')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sample Trajectories of the Noisy Duffing Oscillator with gEDMD Approximation")
plt.legend()
plt.grid()
plt.show()