import torch
import torch.nn.functional as F


def psudo_hober_loss(x, y, c):
    return torch.sqrt(((x - y) ** 2) + (c ** 2)) - c


def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.size(0)
    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def generate_gradual_uniform_vector(size, high_radius=(0.9, 1.0), low_radius=(0.0, 0.5)):
    assert size % 2 == 0, "Size must be even to split into two equal halves."

    # Half for high radius, half for low radius
    half = size // 2

    # Sample angles uniformly in [0, 2π]
    theta_high = torch.rand(half) * 2 * torch.pi
    theta_low = torch.rand(half) * 2 * torch.pi

    # Sample radii in the desired ranges
    r_high = torch.rand(half) * (high_radius[1] - high_radius[0]) + high_radius[0]
    r_low = torch.rand(half) * (low_radius[1] - low_radius[0]) + low_radius[0]

    # Convert to complex numbers using polar form: r * (cosθ + i·sinθ)
    real_high = r_high * torch.cos(theta_high)
    imag_high = r_high * torch.sin(theta_high)

    real_low = r_low * torch.cos(theta_low)
    imag_low = r_low * torch.sin(theta_low)

    # Concatenate and form the complex tensor
    real = torch.cat([real_high, real_low], dim=0)
    imag = torch.cat([imag_high, imag_low], dim=0)
    uniform_vec = torch.complex(real, imag)

    return uniform_vec
