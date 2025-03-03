import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define parameters
alpha = 1
beta = -1
delta = 0.5
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
# initial_conditions = np.random.uniform(-2.5, 2.5, (N_trajectories, 2))
initial_conditions = np.random.randn(N_trajectories, 2)
trajectories = []

for ic in initial_conditions:
    x1_traj, x2_traj = [ic[0]], [ic[1]]
    dt = t_eval[1] - t_eval[0]

    for t in t_eval[:-1]:
        noise = np.random.normal(0, np.sqrt(dt))  # Brownian motion
        state = np.array([x1_traj[-1], x2_traj[-1]])
        dx = duffing_sde(t, state, noise) * dt
        x1_traj.append(state[0] + dx[0])
        x2_traj.append(state[1] + dx[1])

    trajectories.append(np.array([x1_traj, x2_traj]))

trajectories = np.array(trajectories)

# Plot sample trajectories
plt.figure(figsize=(8, 6))
for i in range(40):  # Plot only 20 trajectories for clarity
    plt.plot(trajectories[i, 0, :], trajectories[i, 1, :], alpha=0.5)
plt.scatter(initial_conditions[:40, 0], initial_conditions[:40, 1], color='red', label='Initial Conditions', alpha=0.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sample Trajectories of the Noisy Duffing Oscillator")
plt.legend()
plt.grid()
plt.show()

