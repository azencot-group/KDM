import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def create_2d_dataset(num_samples=1000):
    """
    Creates a 2D synthetic dataset with points sampled from a mixture of Gaussians.

    Args:
        num_samples (int): Number of samples to generate.

    Returns:
        torch.Tensor: Generated 2D points of shape (num_samples, 2).
    """
    centers = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]])
    num_centers = centers.shape[0]
    samples_per_center = num_samples // num_centers

    data = []
    for center in centers:
        samples = np.random.randn(samples_per_center, 2) * 0.5 + center
        data.append(samples)

    data = np.vstack(data)
    np.random.shuffle(data)
    return torch.tensor(data, dtype=torch.float32)

class ScoreModel(nn.Module):
    """
    A simple feedforward neural network to learn the score function.
    """
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    """
    Linear schedule for beta values over the timesteps.

    Args:
        timesteps (int): Number of timesteps.
        start (float): Starting value of beta.
        end (float): Ending value of beta.

    Returns:
        torch.Tensor: Beta values of shape (timesteps,).
    """
    return torch.linspace(start, end, timesteps)

def train_diffusion_score_model(data, epochs=1000, batch_size=128, lr=1e-3, timesteps=100):
    """
    Trains a score model using diffusion.

    Args:
        data (torch.Tensor): 2D dataset of shape (num_samples, 2).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        lr (float): Learning rate for the optimizer.
        timesteps (int): Number of diffusion timesteps.

    Returns:
        ScoreModel: Trained score model.
    """
    dataloader = DataLoader(TensorDataset(data), batch_size=batch_size, shuffle=True)
    model = ScoreModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            x0 = batch[0]
            t = torch.randint(0, timesteps, (x0.shape[0],))
            noise = torch.randn_like(x0)

            sqrt_alpha = sqrt_alphas_cumprod[t].unsqueeze(1)
            sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)

            xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise

            predicted_noise = model(xt)
            loss = nn.MSELoss()(predicted_noise, noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f}")

    return model

def plot_score_field(score_model, grid_range=(-4, 4), grid_size=20):
    """
    Plots the learned score field by the score model.

    Args:
        score_model (ScoreModel): Trained score model.
        grid_range (tuple): Range of the grid in both x and y directions (min, max).
        grid_size (int): Number of points in the grid along each axis.
    """
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    points = torch.tensor(np.c_[X.ravel(), Y.ravel()], dtype=torch.float32)
    with torch.no_grad():
        scores = score_model(points).numpy()

    U = scores[:, 0].reshape(grid_size, grid_size)
    V = scores[:, 1].reshape(grid_size, grid_size)

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, color="blue", scale_units="xy", scale=2)
    plt.title("Learned Score Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Generate 2D synthetic dataset
    num_samples = 1000
    data = create_2d_dataset(num_samples)

    # Train score model using diffusion
    trained_model = train_diffusion_score_model(data, epochs=1000, batch_size=128, lr=1e-3, timesteps=100)

    # Plot the learned score field
    plot_score_field(trained_model, grid_range=(-4, 4), grid_size=20)
