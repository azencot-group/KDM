from datetime import datetime
from pathlib import Path
import logging
from matplotlib import pyplot as plt

import torch

from koopman_distillation.evaluation.fid import translate_to_image_format


def get_workdir(prefix_path, exp):
    workdir = f'{prefix_path}/{exp}/{datetime.now().strftime("%Y_%m_%d_%H_%M_%S")}'
    return workdir


def create_workdir(args):
    workdir = get_workdir(args.output_prefix_path, args.experiment_name)
    Path(workdir).mkdir(parents=True, exist_ok=True)
    torch.save(args, f'{workdir}/args.pth')
    args.workdir = workdir

    return workdir


def log_config_and_tags(args, logger, name):
    logger.log_name_params('config/hyperparameters', vars(args))
    logger.log_name_params('config/name', name)
    logger.add_tags(args.tags)
    logger.add_tags([args.dataset])


def print_model_params(logger, model):
    params_num = sum(param.numel() for param in model.parameters())
    logging.info("number of model parameters: {}".format(params_num))
    logger.log_name_params('config/params_num', params_num)


def plot_samples(logger, model, batch_size, device, data_shape, output_dir, iteration):
    x0_sample = model.sample(batch_size, device, data_shape=data_shape)
    if data_shape == (2,):
        fig = plt.figure(figsize=(7, 9))
        x0_sample = x0_sample[0].detach().cpu().numpy()
        # plot all the points in the batch
        plt.scatter(x0_sample[:, 0], x0_sample[:, 1], c='r', s=1)
        logger.log('scatter_2d', fig)
    else:
        for i in range(3):
            fig = plt.figure(figsize=(7, 9))
            # plot the generated image
            img = translate_to_image_format(x0_sample[0][i].unsqueeze(dim=0))[0]
            plt.imshow(img)
            logger.log(f'{output_dir}/image_{i}', fig)
