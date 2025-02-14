import numpy as np
from scipy.integrate import solve_ivp


def van_der_pol(t, y, mu=1.0, sigma=0.1):
    x, v = y
    dxdt = v
    dvdt = mu * (1 - x ** 2) * v - x + sigma * np.random.randn()
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


def generate_series_data_vdp(n_trajectories=1000, T=10, dt=0.1):
    trajectories, t_eval, initial_conditions = simulate_vdp(n_trajectories=n_trajectories, T=T, dt=dt)
    return trajectories, t_eval, initial_conditions
