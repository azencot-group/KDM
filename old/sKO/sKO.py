import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression

# Define parameters
alpha = 1
beta = -1
delta = 0.25
epsilon = 0.0025
T = 10  # Simulation time
N_trajectories = 400  # Number of independent trajectories


def duffing_sde(t, state, noise):
    """ Stochastic Duffing oscillator system """
    x1, x2 = state
    dx1 = x2
    dx2 = -delta * x2 - x1 * (beta + alpha * x1 ** 2) + np.sqrt(2 * epsilon) * noise
    return np.array([dx1, dx2])


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

        # if not none append to koopman data and the maximum value is 10
        if not np.isnan(state).any() and not np.isnan(dx).any() and np.max(state) < 10 and np.max(state + dx) < 10:
            koopman_data_X.append(state)
            koopman_data_Y.append(state + dx)

    # append only if not nan
    if not np.isnan(x1_traj).any() and not np.isnan(x2_traj).any() and np.max(x1_traj) < 10 and np.max(x2_traj) < 10:
        trajectories.append(np.array([x1_traj, x2_traj]))

trajectories = np.array(trajectories)

# Koopman Operator Approximation
koopman_data_X = np.array(koopman_data_X)
koopman_data_Y = np.array(koopman_data_Y)
koopman_model = LinearRegression()
koopman_model.fit(koopman_data_X, koopman_data_Y)
K_matrix = koopman_model.coef_

# Generate approximated Koopman trajectories
koopman_trajectories = []
for ic in initial_conditions[:20]:
    x_traj = [ic]
    for _ in range(len(t_eval) - 1):
        x_traj.append(K_matrix @ x_traj[-1])
    koopman_trajectories.append(np.array(x_traj).T)

koopman_trajectories = np.array(koopman_trajectories)

# Plot sample trajectories
plt.figure(figsize=(8, 6))
for i in range(3):  # Plot only 20 trajectories for clarity
    plt.plot(trajectories[i, 0, :], trajectories[i, 1, :], alpha=0.5, label='SDE' if i == 0 else "")
    plt.plot(koopman_trajectories[i, 0, :], koopman_trajectories[i, 1, :], '--', alpha=0.5,
             label='Koopman Approx' if i == 0 else "")
plt.scatter(initial_conditions[:3, 0], initial_conditions[:3, 1], color='red', label='Initial Conditions')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sample Trajectories of the Noisy Duffing Oscillator with Koopman Approximation")
plt.legend()
plt.grid()
plt.show()

