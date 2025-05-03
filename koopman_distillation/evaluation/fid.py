import torch

import numpy as np
from pytorch_image_generation_metrics import get_inception_score_and_fid



def translate_to_image_format(images):
    images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return images_np


def sample_and_calculate_fid_and_is(model, data_shape, num_samples, device, batch_size, epoch, image_dir, cond=False):
    i = 0
    all_images = []
    while True:
        if cond:
            labels = torch.eye(model.label_dim, device=device)[
                torch.randint(model.label_dim, size=[batch_size], device=device)]
        else:
            labels = None
        x0_sample = model.sample(batch_size, device, data_shape, labels=labels)
        images = x0_sample[0].detach().cpu()
        for img in images:
            all_images.append((img * 127.5 + 128).clip(0, 255).to(torch.uint8).float().div(255).numpy())
            i += 1
            if i >= num_samples:
                break
        if i >= num_samples:
            break

    all_images = np.stack(all_images, axis=0)

    # Compute FID & IS
    (IS, IS_std), FID = get_inception_score_and_fid(torch.tensor(all_images),
                                                    '/cs/cs_groups/azencot_group/functional_diffusion/cifar10-32x32.npy')

    return IS, FID
