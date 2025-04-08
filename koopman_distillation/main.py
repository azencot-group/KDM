import sys
import os
import random
import socket
from datetime import timedelta
import torch
import torch.multiprocessing as mp
from torch.distributed import (init_process_group, destroy_process_group, all_gather_object, all_gather,
                               barrier)

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from koopman_distillation.utils.names import CondType
from koopman_distillation.utils.loggers import CompositeLogger, NeptuneLogger, TqdmLogger
from koopman_distillation.configs.config_router import get_configs
from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.trainer.trainer import TrainLoop
from koopman_distillation.utils.loading_models import create_distillation_model, loading_pre_trained_weights
from koopman_distillation.utils.loggers.logging import create_workdir, log_config_and_tags, print_model_params
from koopman_distillation.utils.dist_lib import is_distributed, is_main_process


def next_free_port(port=12355, max_port=65535):
    rand_offset = random.randint(0, 1000)
    port += rand_offset
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')


def ddp_setup(rank, world_size, port="12355"):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = port
    init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(minutes=60))
    torch.cuda.set_device(rank)


def main_rank(rank, args, port):
    args.rank = rank
    ddp_setup(args.rank, args.gpu_num, port)
    main(args)
    destroy_process_group()


def main(args):
    # --- create logging --- #
    with CompositeLogger(loggers=[TqdmLogger(no_plot=True),
                                  NeptuneLogger(stdout=False, stderr=False, rank=args.rank,
                                                project=args.neptune_projects)]
            , rank=args.rank) if args.neptune else TqdmLogger(rank=args.rank) as logger:

        if is_distributed(args):
            device = args.rank
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if is_main_process(args):
            # --- create working directory and save the run configs for reproduction--- #
            workdir = create_workdir(args)
            args.workdir = workdir
            log_config_and_tags(args, logger, name='distillation')
            logger.info(f"working directory: {workdir}")

        num_accumulation_rounds = args.nar

        # Setup data:
        local_batch_size = args.batch_size // num_accumulation_rounds // args.gpu_num

        # --- create models --- #
        logger.info("creating observables encoder model...")  # create the main multimodal diffusion bridge framework
        dsm = create_distillation_model(args.distillation_model, args)
        dsm = dsm.to(device)
        print_model_params(logger, dsm)

        # --- load pre-trained models if needed --- #
        loading_pre_trained_weights(dsm)

        # --- load data --- #
        logger.info("creating data loader...")
        train_data, test_data = load_data(
            args=args,
            dataset=args.dataset,
            dataset_path=args.datapath,
            dataset_path_test=args.datapath_test,
            batch_size=local_batch_size,
            num_workers=args.num_workers,
        )

        logger.info("training...")
        TrainLoop(
            args=args,
            model=dsm,
            train_data=train_data,
            test_data=test_data,
            iterations=args.iterations,
            lr=args.lr,
            print_every=args.print_every,
            data_shape=args.data_shape,
            batch_size=local_batch_size,
            device=device,
            logger=logger,
            num_accumulation_rounds=num_accumulation_rounds,
            advers=args.advers,
            cond=args.cond_type != CondType.Uncond,
        ).train()


if __name__ == "__main__":
    args = get_configs()
    args.iterations = ((800_001) * 128) // args.batch_size
    args.print_every = args.iterations // 4_000

    if args.gpu_num > 1 or args.gpu_num < 1:
        world_size = args.gpu_num if args.gpu_num != -1 else torch.cuda.device_count()
        args.gpu_num = world_size
        port = str(next_free_port())
        mp.spawn(main_rank, args=(args, port), nprocs=world_size)
    else:
        args.rank = 0
        main(args)
