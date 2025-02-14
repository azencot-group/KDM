import numpy as np
import matplotlib.pyplot as plt
import torch.nn
import time

from our_utils import linear_beta_schedule


def create_dataset(num_samples):
    """
    Creates a dataset of rectangles with specific pixel configurations.

    Args:
        num_samples (int): The number of samples to generate.

    Returns:
        np.ndarray: A dataset of shape (num_samples, 3, 4) containing the generated rectangles.
        np.ndarray: A corresponding array of shape (num_samples, 2) containing the chosen Gaussian means for the middle pixels.
    """
    # Define the possible Gaussian means
    gaussian_means = np.array([[1, 1], [-1, 1], [1, -1], [-1, -1]])
    std_dev = 0.1  # Standard deviation for the Gaussian distribution

    # Initialize the dataset
    dataset = np.zeros((num_samples, 3, 4))
    labels = np.zeros((num_samples, 2))

    for i in range(num_samples):
        # Randomly select one of the Gaussian means
        chosen_mean = gaussian_means[np.random.choice(len(gaussian_means))]
        labels[i] = chosen_mean

        # Add Gaussian noise with standard deviation 1 to the mean
        noise = np.random.normal(loc=0, scale=std_dev, size=2)
        noisy_values = chosen_mean + noise

        # Assign the noisy values to the middle pixels
        dataset[i, 1, 1] = noisy_values[0]
        dataset[i, 1, 2] = noisy_values[1]

    return dataset, labels


def fill_with_correlated_values(dataset, noise_factor=0.1):
    """
    Modifies each pixel in the dataset by filling it with a value that is linearly or non-linearly
    correlated with one of the middle pixels that has a non-zero value, with added Gaussian noise.

    Args:
        dataset (np.ndarray): A dataset of shape (num_samples, 3, 4).
        noise_factor (float): A factor controlling the magnitude of Gaussian noise added.

    Returns:
        np.ndarray: The modified dataset.
    """
    modified_dataset = np.copy(dataset)
    num_samples = dataset.shape[0]

    for i in range(num_samples):
        middle_pixel_1 = dataset[i, 1, 1]
        middle_pixel_2 = dataset[i, 1, 2]

        for row in range(3):
            for col in range(4):
                if row == 1 and (col == 1 or col == 2):
                    continue  # Skip the middle pixels

                # Choose a linear or non-linear correlation with the middle pixels
                if np.random.rand() > 0.5:
                    correlated_value = 2 * middle_pixel_1 + 0.5 * middle_pixel_2
                else:
                    correlated_value = np.sin(middle_pixel_1) + np.cos(middle_pixel_2)

                # Add Gaussian noise
                noise = np.random.normal(loc=0, scale=noise_factor)
                modified_dataset[i, row, col] = correlated_value + noise

    return modified_dataset


def plot_data_distribution(dataset):
    """
    Plots the distribution of the data points in the middle pixels of the dataset.

    Args:
        dataset (np.ndarray): A dataset of shape (num_samples, 3, 4).
    """
    middle_pixels = dataset[:, 1, 1:3]

    plt.figure(figsize=(6, 6))
    plt.scatter(middle_pixels[:, 0], middle_pixels[:, 1], alpha=0.6, label="Data Points")

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.title("Data Distribution in Middle Pixels")
    plt.xlabel("First Dimension")
    plt.ylabel("Second Dimension")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # ################## data ################### #
    # Example usage:
    num_samples = 1000
    irregular_data, labels = create_dataset(num_samples)
    regular_data = fill_with_correlated_values(irregular_data, noise_factor=0)
    print("Generated Data Shape:", irregular_data.shape)
    print("Generated Labels Shape:", labels.shape)

    # --- exhibit1 - Plot the data distribution of the Gaussians --- #
    plot_data_distribution(irregular_data)

    # # --- exhibit2 - plot a data point, add frame to the plot --- #
    # plt.imshow(irregular_data[0], cmap='gray', interpolation='none')
    # plt.gca().invert_yaxis()
    # plt.gca().set_frame_on(False)
    # # scale color
    # plt.colorbar()
    # plt.title("Example Data Point Irregular")
    # plt.axis('off')
    # plt.show()
    #
    # # --- exhibit3 - plot a data point, add frame to the plot --- #
    # plt.imshow(regular_data[0], cmap='gray', interpolation='none')
    # plt.gca().invert_yaxis()
    # plt.gca().set_frame_on(False)
    # # scale color
    # plt.colorbar()
    # plt.title("Example Data Point Regular")
    # plt.axis('off')
    # plt.show()

    # ################## model ################### #
    epochs = 2000
    batch_size = 256
    lr = 0.0001
    print_every = 10

    mask = torch.zeros(3, 4)
    mask[1, 1] = 1
    mask[1, 2] = 1
    mask = mask[None, None, :, :]

    score_model = torch.nn.Sequential(torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(3, 4)),
                                      torch.nn.ReLU())
    optim = torch.optim.Adam(score_model.parameters(), lr=lr)

    beta_schedule_fn = linear_beta_schedule
    betas = beta_schedule_fn(1000)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).float()
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).float()

    # ################## training ################### #

    data_loader_train = torch.utils.data.DataLoader(irregular_data[:800], batch_size=batch_size, shuffle=True)
    data_loader_test = torch.utils.data.DataLoader(irregular_data[800:], batch_size=batch_size, shuffle=True)

    start_time = time.time()
    for epoch in range(epochs):

        for i, data in enumerate(data_loader_train):
            # zero the parameter gradients
            optim.zero_grad()

            x0 = data.float().unsqueeze(1)
            noise = torch.randn_like(x0)
            t = torch.randint(0, 1000, (x0.shape[0],)).to(x0.device)

            # forward pass
            alphas = sqrt_alphas_cumprod.to(x0.device)[t][:, None, None, None]
            betas = sqrt_one_minus_alphas_cumprod.to(x0.device)[t][:, None, None, None]

            xt = (x0 * (1 - mask)) + (alphas * x0 * mask) + (betas * noise * mask)

            score = score_model(xt)

            loss = torch.nn.functional.mse_loss(score.reshape(xt.shape[0], -1), noise[:, :, [1, 1], [1, 2]].squeeze())
            loss.backward()
            optim.step()  # update

        # if (epoch + 1) % print_every == 0:
        #     elapsed = time.time() - start_time
        #     print('| iter {:6d} | {:5.2f} ms/step | loss {:8.10f} '
        #           .format(epoch + 1, elapsed * 1000 / print_every, loss.item()))
        loss_total = 0
        for i, data in enumerate(data_loader_test):
            with torch.no_grad():
                x0 = data.float().unsqueeze(1)
                noise = torch.randn_like(x0)
                t = torch.randint(0, 1000, (x0.shape[0],)).to(x0.device)

                # forward pass
                alphas = sqrt_alphas_cumprod.to(x0.device)[t][:, None, None, None]
                betas = sqrt_one_minus_alphas_cumprod.to(x0.device)[t][:, None, None, None]

                xt = (x0 * (1 - mask)) + (alphas * x0 * mask) + (betas * noise * mask)

                score = score_model(xt)

                loss = torch.nn.functional.mse_loss(score.reshape(xt.shape[0], -1),
                                                    noise[:, :, [1, 1], [1, 2]].squeeze())
                loss_total = + loss

        if (epoch + 1) % print_every == 0:
            elapsed = time.time() - start_time
            print('| iter {:6d} | {:5.2f} ms/step | loss {:8.10f} '
                  .format(epoch + 1, elapsed * 1000 / print_every, loss_total.item() / (i + 1)))


    def plot_score_function(score_function, grid_range=(-3, 3), grid_size=50):
        """
        Plots the score function of a 2D diffusion model as a vector field.

        Args:
            score_function (callable): A function that takes a 2D input (x, y) and returns a 2D score (dx, dy).
            grid_range (tuple): The range of the grid in both x and y directions (min, max).
            grid_size (int): The number of points in the grid along each axis.
        """
        x = np.linspace(grid_range[0], grid_range[1], grid_size)
        y = np.linspace(grid_range[0], grid_range[1], grid_size)
        X, Y = np.meshgrid(x, y)

        # Compute the scores for each grid point
        U = np.zeros_like(X)
        V = np.zeros_like(Y)
        for i in range(grid_size):
            for j in range(grid_size):
                x = torch.zeros((1, 1, 3, 4))
                x[:, :, 1, 1] = X[i, j]
                x[:, :, 1, 2] = Y[i, j]
                score = score_function(torch.tensor(x))
                U[i, j], V[i, j] = score.reshape(-1)

        # Plot the vector field
        plt.figure(figsize=(8, 8))
        plt.quiver(X, Y, U, V, color="blue", scale_units="xy", scale=4)
        plt.title("Score Function Vector Field")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.grid(True)
        plt.show()


    # Plot the score function
    plot_score_function(score_model, grid_range=(-3, 3), grid_size=30)
