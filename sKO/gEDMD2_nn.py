import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from sklearn.linear_model import LinearRegression
import torch
import torch.nn as nn
import torch.optim as optim

# Define parameters
alpha = 1
beta = -1
delta = 0.5
epsilon = 0.0025
T = 10  # Simulation time
N_trajectories = 1000  # Number of independent trajectories


# Define the neural network for lifting (observables) and inverse mapping
class LiftingNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=66):
        super(LiftingNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


class InverseNN(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=128, output_dim=2):
        super(InverseNN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Initialize neural networks
lifting_nn = LiftingNN()
inverse_nn = InverseNN()

# Define optimizer and loss function
optimizer_lift = optim.Adam(lifting_nn.parameters(), lr=0.001)
optimizer_inverse = optim.Adam(inverse_nn.parameters(), lr=0.001)
loss_fn = nn.MSELoss()


def duffing_sde(t, state, noise, mu=1, sigma=0.1):
    """ Stochastic Duffing oscillator system """
    x, v = state
    dxdt = v
    dvdt = mu * (1 - x**2) * v - x + sigma * np.random.randn()
    return np.array([dxdt, dvdt])


# Generate trajectories
t_eval = np.linspace(0, T, 100)  # Time points
initial_conditions = np.random.uniform(-2.5, 2.5, (N_trajectories, 2))
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
trajectories_flat = torch.tensor(trajectories.reshape(-1, 2)).float()

epochs = 50
for epoch in range(epochs):
    optimizer_inverse.zero_grad()
    optimizer_lift.zero_grad()
    predictions = inverse_nn(lifting_nn(trajectories_flat))
    loss = loss_fn(predictions, trajectories_flat)
    loss.backward()
    # print loss
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item()}")
    optimizer_inverse.step()
    optimizer_lift.step()


# aggerate all koopman projections
koopman_data_X = []
koopman_data_Y = []
for ic in initial_conditions:
    x1_traj, x2_traj = [ic[0]], [ic[1]]
    dt = t_eval[1] - t_eval[0]

    for t in t_eval[:-1]:
        noise = np.random.normal(0, np.sqrt(dt))  # Brownian motion
        state = np.array([x1_traj[-1], x2_traj[-1]])
        dx = duffing_sde(t, state, noise) * dt
        koopman_data_X.append(lifting_nn(torch.tensor(state, dtype=torch.float32)).detach().numpy())
        koopman_data_Y.append(lifting_nn(torch.tensor(state + dx, dtype=torch.float32)).detach().numpy())

# Koopman Operator Approximation using gEDMD
koopman_data_X = np.array(koopman_data_X)
koopman_data_Y = np.array(koopman_data_Y)
koopman_model = LinearRegression(fit_intercept=False)
koopman_model.fit(koopman_data_X, koopman_data_Y)
K_matrix = koopman_model.coef_.T  # Ensure square Koopman matrix

# Eigenfunction decomposition using eigendecomposition
eigenvalues, eigenvectors = np.linalg.eig(K_matrix)

# Generate approximated Koopman trajectories using eigenfunctions
gedmd_trajectories = []
for ic in initial_conditions[:20]:
    x_traj = [lifting_nn(torch.tensor(ic, dtype=torch.float32)).detach().numpy()]
    for _ in range(len(t_eval) - 1):
        x_next = K_matrix @ x_traj[-1]  # Ensure proper dimension match
        x_traj.append(x_next)
    gedmd_trajectories.append(np.array(x_traj).T)

gedmd_trajectories = np.array(gedmd_trajectories)

# Convert Koopman-based trajectory back to original space
inverse_trajectories = np.array(
    [inverse_nn(torch.tensor(y, dtype=torch.float32).T).detach().numpy().T for y in gedmd_trajectories])

# Plot sample trajectories
plt.figure(figsize=(8, 6))
for i in range(5):  # Plot only 20 trajectories for clarity
    plt.plot(trajectories[i, 0, :], trajectories[i, 1, :], alpha=0.5, label='SDE' if i == 0 else "")
    plt.plot(inverse_trajectories[i, 0, :], inverse_trajectories[i, 1, :], '--', alpha=0.5,
             label='Inverse gEDMD Approx' if i == 0 else "")
plt.scatter(initial_conditions[:5, 0], initial_conditions[:5, 1], color='red', label='Initial Conditions')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Sample Trajectories of the Noisy Duffing Oscillator with gEDMD Approximation Using Neural Networks")
plt.legend()
plt.grid()
plt.show()