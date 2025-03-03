# --------- Imports and init device ---------- #
import time
import torch

from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper
import numpy as np

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm


# To avoide meshgrid warning
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

if torch.cuda.is_available():
    device = 'cuda:0'
    print('Using gpu')
else:
    device = 'cpu'
    print('Using cpu.')

torch.manual_seed(42)


# --------- Dataset ---------- #

def inf_train_gen(batch_size: int = 200, device: str = "cpu"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data.float()




# --------- Model ---------- #
# Activation class
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(x) * x


# Model class
class MLP(nn.Module):
    def __init__(self, input_dim: int = 2, time_dim: int = 1, hidden_dim: int = 128):
        super().__init__()

        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            nn.Linear(input_dim + time_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, hidden_dim),
            Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        sz = x.size()
        x = x.reshape(-1, self.input_dim)
        t = t.reshape(-1, self.time_dim).float()

        t = t.reshape(-1, 1).expand(x.shape[0], 1)
        h = torch.cat([x, t], dim=1)
        output = self.main(h)

        return output.reshape(*sz)


# --------- Training ---------- #

# training arguments
lr = 0.001
batch_size = 4096
iterations = 20001
print_every = 2000
hidden_dim = 512

# velocity field model init
vf = MLP(input_dim=2, time_dim=1, hidden_dim=hidden_dim).to(device)

# instantiate an affine path object
path = AffineProbPath(scheduler=CondOTScheduler())

# init optimizer
optim = torch.optim.Adam(vf.parameters(), lr=lr)

# train
# start_time = time.time()
# for i in range(iterations):
#     optim.zero_grad()
#
#     # sample data (user's responsibility): in this case, (X_0,X_1) ~ pi(X_0,X_1) = N(X_0|0,I)q(X_1)
#     x_1 = inf_train_gen(batch_size=batch_size, device=device)  # sample data
#     x_0 = torch.randn_like(x_1).to(device)
#
#     # sample time (user's responsibility)
#     t = torch.rand(x_1.shape[0]).to(device)
#
#     # sample probability path
#     path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
#
#     # flow matching l2 loss
#     loss = torch.pow(vf(path_sample.x_t, path_sample.t) - path_sample.dx_t, 2).mean()
#
#     # optimizer step
#     loss.backward()  # backward
#     optim.step()  # update
#
#     # log loss
#     if (i + 1) % print_every == 0:
#         elapsed = time.time() - start_time
#         print('| iter {:6d} | {:5.2f} ms/step | loss {:8.3f} '
#               .format(i + 1, elapsed * 1000 / print_every, loss.item()))
#         start_time = time.time()

# save model
# torch.save(vf.state_dict(), 'vf.pth')
# print('Model saved as vf.pth')

# load model
vf.load_state_dict(torch.load('vf.pth'))


# --------- sample and visualize ---------- #
class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model(x, t)

wrapped_vf = WrappedModel(vf)


# step size for ode solver
step_size = 0.05

norm = cm.colors.Normalize(vmax=50, vmin=0)

batch_size = 50000  # batch size
eps_time = 1e-2
T = torch.linspace(0,1,10)  # sample times
T = T.to(device=device)

x_init = torch.randn((batch_size, 2), dtype=torch.float32, device=device)
solver = ODESolver(velocity_model=wrapped_vf)  # create an ODESolver class
sol = solver.sample(time_grid=T, x_init=x_init, method='midpoint', step_size=step_size, return_intermediates=True)  # sample from the model

# plot
sol = sol.cpu().numpy()
T = T.cpu()

fig, axs = plt.subplots(1, 10, figsize=(20, 20))

for i in range(10):
    H = axs[i].hist2d(sol[i, :, 0], sol[i, :, 1], 300, range=((-5, 5), (-5, 5)))

    cmin = 0.0
    cmax = torch.quantile(torch.from_numpy(H[0]), 0.99).item()

    norm = cm.colors.Normalize(vmax=cmax, vmin=cmin)

    _ = axs[i].hist2d(sol[i, :, 0], sol[i, :, 1], 300, range=((-5, 5), (-5, 5)), norm=norm)

    axs[i].set_aspect('equal')
    axs[i].axis('off')
    axs[i].set_title('t= %.2f' % (T[i]))

plt.tight_layout()
plt.show()
#
# # save sol
# np.save('sol.npy', sol)

# load sol
sol = np.load('sol.npy')


# plot in colors each of the square
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate a simulated dataset (replace this with your numpy array `points`)
# points = np.load('your_points_file.npy') # Load your numpy array here
points = sol[-1] # Simulated for demonstration

# Step 1: Use clustering to identify the distinct squares (assuming 8 squares)
n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(points)

# Step 2: Plot the points, color-coded by cluster
points = sol[0]
plt.figure(figsize=(8, 8))
for cluster_id in range(n_clusters):
    cluster_points = points[labels == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Square {cluster_id + 1}', alpha=0.6)

# Mark the cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color='red', marker='x', s=100, label='Centers')

# Formatting the plot
plt.legend()
plt.title("Points Grouped into Squares")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.grid(True)
plt.show()
