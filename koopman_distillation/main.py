import sys
import torch

sys.path.append('.')
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../../')

from koopman_distillation.utils.names import CondType
from koopman_distillation.utils.loggers import CompositeLogger, NeptuneLogger, PrintLogger
from koopman_distillation.configs.config_router import get_configs
from koopman_distillation.data.data_loading.data_loaders import load_data
from koopman_distillation.trainer.trainer import TrainLoop
from koopman_distillation.utils.loading_models import create_distillation_model, loading_pre_trained_weights
from koopman_distillation.utils.loggers.logging import create_workdir, log_config_and_tags, print_model_params


def main(args):
    # --- create logging --- #
    with CompositeLogger([NeptuneLogger(project=args.neptune_projects)]) if args.neptune \
            else PrintLogger() as logger:
        # --- create working directory and save the run configs for reproduction--- #
        workdir = create_workdir(args)
        log_config_and_tags(args, logger, name='distillation')
        logger.info(f"working directory: {workdir}")

        # --- create models --- #
        logger.info("creating observables encoder model...")  # create the main multimodal diffusion bridge framework
        dsm = create_distillation_model(args.distillation_model, args)
        dsm = dsm.cuda()
        print_model_params(logger, dsm)

        # --- load pre-trained models if needed --- #
        loading_pre_trained_weights(dsm)

        # --- load data --- #
        logger.info("creating data loader...")
        train_data, test_data = load_data(
            dataset=args.dataset,
            dataset_path=args.datapath,
            dataset_path_test=args.datapath_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

        logger.info("training...")
        TrainLoop(
            model=dsm,
            train_data=train_data,
            test_data=test_data,
            iterations=args.iterations,
            lr=args.lr,
            print_every=args.print_every,
            data_shape=args.data_shape,
            batch_size=args.batch_size,
            output_dir=workdir,
            device=torch.device('cuda'),
            logger=logger,
            ema_rate=args.ema_rate,
            advers=args.advers,
            cond=args.cond_type != CondType.Uncond,
        ).train()


if __name__ == "__main__":
    arguments = get_configs()
    main(arguments)
