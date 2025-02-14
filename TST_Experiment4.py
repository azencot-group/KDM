import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def create_augmented_dataset(data_images):
    """
    Creates an augmented dataset where non-middle pixels contain linear or non-linear
    functions of the middle pixel values.

    Args:
        data_images (torch.Tensor): Original dataset with 3x4 images of shape (num_samples, 1, 3, 4).

    Returns:
        torch.Tensor: Augmented dataset of shape (num_samples, 1, 3, 4).
    """
    augmented_images = data_images.clone()
    x_middle = data_images[:, 0, 1, 1]
    y_middle = data_images[:, 0, 1, 2]

    # Example augmentation: Fill remaining pixels with simple functions of middle pixels
    augmented_images[:, 0, 0, 0] = x_middle + y_middle
    augmented_images[:, 0, 0, 1] = x_middle + 1
    augmented_images[:, 0, 0, 2] = y_middle - 2
    augmented_images[:, 0, 0, 3] = x_middle - y_middle

    augmented_images[:, 0, 1, 0] = torch.sin(y_middle)
    augmented_images[:, 0, 1, 3] = torch.cos(y_middle)

    augmented_images[:, 0, 2, 0] = torch.sin(x_middle)
    augmented_images[:, 0, 2, 1] = x_middle ** 2
    augmented_images[:, 0, 2, 2] = torch.cos(x_middle)
    augmented_images[:, 0, 2, 3] = y_middle ** 2

    return augmented_images


def create_2d_dataset(num_samples=1000):
    centers = np.array([[2, 2], [-2, -2], [2, -2], [-2, 2]])
    num_centers = centers.shape[0]
    samples_per_center = num_samples // num_centers

    data = []
    for center in centers:
        samples = np.random.randn(samples_per_center, 2) * 0.5 + center
        data.append(samples)

    data = np.vstack(data)
    np.random.shuffle(data)
    data_images = np.zeros((num_samples, 1, 3, 4))
    data_images[:, 0, 1, 1] = data[:, 0]
    data_images[:, 0, 1, 2] = data[:, 1]

    return torch.tensor(data, dtype=torch.float32), torch.tensor(data_images, dtype=torch.float32)


class ScoreModel(nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 4), padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=(3, 4), padding=0),
        )

    def forward(self, x):
        return self.network(x)


def linear_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)


def train_diffusion_score_model(data_images, epochs=1000, batch_size=128, lr=1e-3, timesteps=100, masked=None):
    dataloader = DataLoader(TensorDataset(data_images), batch_size=batch_size, shuffle=True)
    model = ScoreModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    betas = linear_beta_schedule(timesteps)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

    for epoch in range(epochs):
        epoch_loss = 0.0
        coor_loss = 0.0
        for batch in dataloader:
            x0 = batch[0]
            t = torch.randint(0, timesteps, (x0.shape[0],))
            noise = torch.randn_like(x0)

            sqrt_alpha = sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_one_minus_alpha = sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

            xt = sqrt_alpha * x0 + sqrt_one_minus_alpha * noise
            if masked is not None:
                # add the real values everywhere but the middle pixel
                xt = xt * masked + x0 * (1 - masked)

            predicted_noise = model(xt)
            if masked is not None:
                loss = nn.MSELoss()(predicted_noise[:, 0, [1, 1], [1, 2]], noise[:, 0, [1, 1], [1, 2]])
            else:
                loss = nn.MSELoss()(predicted_noise, noise)

            with torch.no_grad():
                coor_loss += nn.MSELoss()(predicted_noise[:, 0, [1, 1], [1, 2]], noise[:, 0, [1, 1], [1, 2]]).item()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.6f} , Loss: {coor_loss / len(dataloader):.6f}")
    return model


def plot_data_and_score_field(score_model, data, grid_range=(-4, 4), grid_size=20):
    x = np.linspace(grid_range[0], grid_range[1], grid_size)
    y = np.linspace(grid_range[0], grid_range[1], grid_size)
    X, Y = np.meshgrid(x, y)

    grid_images = np.zeros((grid_size * grid_size, 1, 3, 4))
    grid_images[:, 0, 1, 1] = X.ravel()
    grid_images[:, 0, 1, 2] = Y.ravel()
    grid_images = torch.tensor(grid_images, dtype=torch.float32)

    with torch.no_grad():
        scores = score_model(grid_images).numpy()

    U = scores[:, 0, 1, 1].reshape(grid_size, grid_size)
    V = scores[:, 0, 1, 2].reshape(grid_size, grid_size)

    plt.figure(figsize=(8, 8))
    plt.scatter(data[:, 0], data[:, 1], color="red", alpha=0.6, label="Data Points")
    plt.quiver(X, Y, U, V, color="blue", scale_units="xy", scale=2, label="Score Field")
    plt.title("Data and Learned Score Field")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_sample_and_kernel(data_images, score_model, plot_data_image=False):
    """
    Plots a single sample from the dataset and the kernels of the model's first Conv2d layer.

    Args:
        data_images (torch.Tensor): 3x4 images of shape (num_samples, 1, 3, 4).
        score_model (ScoreModel): Trained score model.
    """
    if plot_data_image:
        # Plot a single sample from the dataset
        sample = data_images[0].numpy().squeeze()
        plt.figure(figsize=(5, 5))
        plt.imshow(sample, cmap="gray", aspect="auto")
        plt.title("Sample from Dataset")
        plt.colorbar()
        # set the grid to be 0 < x < 3 and y 0 < y < 4 and a gap of 1
        plt.xticks(np.arange(0, 4, 1))
        plt.yticks(np.arange(0, 3, 1))
        plt.grid()
        plt.show()

    # Plot the kernels of the first Conv2d layer
    kernels = score_model.network[0].weight.data.numpy()
    num_kernels = kernels.shape[0]
    plt.figure(figsize=(32, 2))
    for i in range(num_kernels):
        plt.subplot(2, num_kernels // 2, i + 1)
        plt.imshow(kernels[i, 0], cmap="gray")
        plt.axis("off")
    plt.suptitle("Kernels of First Conv2d Layer")
    plt.colorbar()
    plt.show()

    # also plot an absolute average of the kernels
    plt.figure(figsize=(5, 5))
    plt.imshow(np.abs(kernels).mean(axis=0)[0], cmap="gray")
    plt.title("Average of Kernels")
    plt.colorbar()
    plt.show()


def plot_average_kernels(score_models, model_names):
    """
    For each model in 'score_models', plot the absolute average of its
    first Conv2D layer's kernels in a separate subplot.

    Parameters
    ----------
    score_models : list
        A list of PyTorch models, each having at least one Conv2D layer
        accessible via score_model.network[0].
    """

    num_models = len(score_models)

    # Create a figure with one row and `num_models` columns
    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 4), squeeze=False)

    for i, model in enumerate(score_models):
        # Extract the kernels from the first Conv2D layer
        kernels = model.network[0].weight.data.numpy()  # shape: [out_channels, in_channels, kH, kW]

        # Compute the absolute average across out_channels & in_channels
        avg_kernel = np.abs(kernels).mean(axis=(0, 1))  # shape: [kH, kW]

        # calculate the average value in the none middle pixels
        copy_kernel = avg_kernel.copy()
        copy_kernel[1, 1] = 0
        copy_kernel[1, 2] = 0
        avg_value = copy_kernel.sum() / 10
        print(f'Average value in the none middle pixels for model {model_names[i]}: {avg_value}')

        # Display the absolute average in its own subplot
        ax = axes[0, i]
        im = ax.imshow(avg_kernel, cmap="gray", aspect="auto")
        # ax.set_title(f"Model {model_names[i]} Avg. Kernel")
        ax.axis("off")

        # Add a colorbar for each subplot
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Add an overall title (optional)
    # plt.suptitle("Absolute Average of First Conv2D Kernels", y=1.02, fontsize=14)

    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt


def plot_combined(data, augmented_data_images, data_images):
    """
    Combines three different plots into one figure with three subplots:
      1) 2D scatter plot of the dataset
      2) Single sample from augmented_data_images
      3) Single sample from data_images
    """

    # Create a figure with 3 subplots horizontally
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # ----- Subplot 1: 2D scatter of the dataset -----
    axs[0].scatter(
        data[:, 0],
        data[:, 1],
        color="red",
        alpha=0.6,
        label="Data Points"
    )
    # axs[0].set_title("2D Dataset Samples")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].legend()
    axs[0].grid(True)

    # ----- Subplot 2: Sample from augmented_data_images -----
    sample_aug = augmented_data_images[0].numpy().squeeze()
    im2 = axs[1].imshow(sample_aug, cmap="gray", aspect="auto")
    # axs[1].set_title("Sample from Augmented Dataset")
    # Set grid lines
    axs[1].set_xticks(np.arange(0, 4, 1))
    axs[1].set_yticks(np.arange(0, 3, 1))
    axs[1].grid(True)

    # ----- Subplot 3: Sample from data_images -----
    sample_orig = data_images[0].numpy().squeeze()
    im3 = axs[2].imshow(sample_orig, cmap="gray", aspect="auto")
    # axs[2].set_title("Sample from Original Dataset")
    # Set grid lines
    axs[2].set_xticks(np.arange(0, 4, 1))
    axs[2].set_yticks(np.arange(0, 3, 1))
    axs[2].grid(True)
    # Add a colorbar for this subplot
    fig.colorbar(im3, ax=axs[2], fraction=0.046, pad=0.04)

    # Tight layout for better spacing
    plt.tight_layout()

    # Save and/or show
    # plt.savefig("combined_plot.pdf", dpi=300)
    plt.show()


if __name__ == "__main__":
    train_mode = True

    # Generate 2D synthetic dataset
    num_samples = 1000
    data, data_images = create_2d_dataset(num_samples)
    augmented_data_images = create_augmented_dataset(data_images)

    # mask
    mask = torch.zeros(1, 3, 4)
    mask[0, 1, 1] = 1
    mask[0, 1, 2] = 1

    # Train score model using diffusion
    if train_mode:
        trained_model_masked = train_diffusion_score_model(data_images, epochs=1000, batch_size=128, lr=1e-3, timesteps=100)
        trained_model = train_diffusion_score_model(data_images, epochs=1000, batch_size=128, lr=1e-3, timesteps=100,
                                                    masked=mask)
        trained_model_aug_masked = train_diffusion_score_model(augmented_data_images, epochs=1000, batch_size=128, lr=1e-3,
                                                               timesteps=100, masked=mask)

        # save the trained models
        torch.save(trained_model_masked, "trained_model_masked.pth")
        torch.save(trained_model, "trained_model.pth")
        torch.save(trained_model_aug_masked, "trained_model_aug_masked.pth")

        # save data
        torch.save(data, "data.pth")
        torch.save(data_images, "data_images.pth")
        torch.save(augmented_data_images, "augmented_data_images.pth")

    else:
        # load data
        data = torch.load("data.pth")
        data_images = torch.load("data_images.pth")
        augmented_data_images = torch.load("augmented_data_images.pth")

        # load the saved models
        trained_model_masked = torch.load("trained_model_masked.pth")
        trained_model = torch.load("trained_model.pth")
        trained_model_aug_masked = torch.load("trained_model_aug_masked.pth")

    # Plot the combined figure
    plot_combined(data, augmented_data_images, data_images)
    plot_average_kernels(list([trained_model_masked, trained_model, trained_model_aug_masked]),
                         ["Unmasked Irregular", "Masked Irregular", "Masked Regular"])
