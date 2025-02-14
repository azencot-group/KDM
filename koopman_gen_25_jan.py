import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader


class KoopmanGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, latent_dim):
        super(KoopmanGenerator, self).__init__()

        # Encoder to map data to latent space
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers)],
            nn.Linear(hidden_dim, latent_dim)
        )

        # Koopman operator in latent space
        self.koopman = nn.Linear(latent_dim, latent_dim, bias=False)  # Approximates evolution

        # Decoder to map latent space back to data space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(num_layers)],
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x, time_steps):
        # Encoding
        z = self.encoder(x)

        # Koopman evolution in latent space
        for _ in range(time_steps):
            z = self.koopman(z)

        # Decoding
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def noise_forward(self, x, noise_schedule):
        """Iteratively add noise to data following a diffusion-like forward process."""
        noise_data = []
        # for t in range(len(noise_schedule)):
        for t in range(1):
            # noise = noise_schedule[t] * torch.randn_like(x)
            noise = noise_schedule[-1] * torch.randn_like(x)
            x = x + noise
            noise_data.append(x.clone())
        return noise_data

    def denoise(self, z, steps, noise_schedule):
        """Iteratively denoise latent variable z using Koopman operator."""
        for t in reversed(range(steps)):
            noise = noise_schedule[t] * torch.randn_like(z)
            z = z - noise  # Remove noise
            z = self.koopman(z)  # Apply Koopman operator to refine
        return z

def train_koopman_model(model, dataloader, optimizer, noise_schedule, num_epochs, device):
    """
    Train the Koopman-inspired generative model with iterative denoising.
    """
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for data in dataloader:
            data = data.to(device).squeeze()

            # Forward process: add noise
            noise_data = model.noise_forward(data, noise_schedule)
            noisy_data = noise_data[-1]  # Most noisy data

            # Training step
            optimizer.zero_grad()

            # Predict the reconstructed data
            reconstructed = model(noisy_data, time_steps=len(noise_schedule))

            # Compute loss between reconstructed and original data
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(dataloader)}")

    return model


def generate_with_model(model, latent_dim, noise_schedule, steps, device):
    """
    Generate new samples with the Koopman-inspired generative model.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Start from Gaussian noise in the data space
        x = torch.randn((1024, 2)).to(device)
        z = model.encoder(x)

        # Iteratively denoise in the latent space
        z_denoised = model.denoise(z, steps, noise_schedule)

        # Decode the final latent representation into data space
        generated_data = model.decoder(z_denoised)

    return generated_data.cpu()


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __len__(self):
        return 10000  # Arbitrary large number for infinite sampling

    def __getitem__(self, index):
        return inf_train_gen(self.batch_size)

# ------- data -------
def inf_train_gen(batch_size: int = 200, device: str = "cuda"):
    x1 = torch.rand(batch_size, device=device) * 4 - 2
    x2_ = torch.rand(batch_size, device=device) - torch.randint(high=2, size=(batch_size,), device=device) * 2
    x2 = x2_ + (torch.floor(x1) % 2)

    data = 1.0 * torch.cat([x1[:, None], x2[:, None]], dim=1) / 0.45

    return data.float()

def get_dataloader(dataset_size=100000, batch_size=4096):
    dataset = CustomDataset(batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    return dataloader



input_dim = 2  # 2D data
hidden_dim = 256
num_layers = 3
latent_dim = hidden_dim
model = KoopmanGenerator(input_dim, hidden_dim, num_layers, latent_dim)
noise_schedule = torch.linspace(0.01, 0.7, steps=10).tolist()

dataloader = get_dataloader(dataset_size=100000, batch_size=4096)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model = train_koopman_model(model, dataloader, optimizer, noise_schedule, num_epochs=10, device="cuda")

generated_data = generate_with_model(model, latent_dim, noise_schedule, steps=10, device="cuda")

# plot the generated data
import matplotlib.pyplot as plt
x0_sample = generated_data.detach().cpu().numpy()
# plot all the points in the batch
plt.scatter(x0_sample[:, 0], x0_sample[:, 1], c='r', s=1)
plt.show()

