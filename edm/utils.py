import torch
import matplotlib.pyplot as plt


def plot_image_sequence(image_list):
    """
    Plots a sequence of images from a list.

    Args:
        image_list (torch.Tensor): Tensor of shape (K, C, H, W) containing K images.
    """
    K, C, H, W = image_list.shape
    fig, axes = plt.subplots(1, K, figsize=(K * 2, 2))

    for i in range(K):
        img = image_list[i].cpu().numpy()
        if C == 1:
            img = img.squeeze(0)  # Grayscale image
            axes[i].imshow(img, cmap='gray')
        else:
            img = img.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)
            axes[i].imshow(img)

        axes[i].axis('off')
        axes[i].set_title(f"Frame {i + 1}")

    plt.show()

# Example usage
# images = torch.randn(5, 3, 64, 64)  # Example tensor with 5 RGB images
# plot_image_sequence(images)
