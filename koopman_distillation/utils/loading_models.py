import copy
import pickle

import torch.nn

from edm.dnnlib.util import open_url
from koopman_distillation.model.koopman_distillator import KoopmanDistillOneStep
from koopman_distillation.model.modules.model_checkerboard import Encoder, Decoder
from koopman_distillation.model.modules.model_cifar10 import OneStepKoopmanCifar10, SongUNet
from koopman_distillation.other_methods.consistency_models.models.consistency_model import ConsistencyModel
from koopman_distillation.other_methods.consistency_models.models.diffusion import KarrasDenoiser
from koopman_distillation.other_methods.consistency_models.models.ema import create_ema_and_scales_fn
from koopman_distillation.other_methods.consistency_models.models.unet import UNetModel
from koopman_distillation.utils.names import DistillationModels, Datasets


def create_distillation_model(model_type: DistillationModels, args):
    if model_type == DistillationModels.OneStepKOD:
        return create_koopman_model(args)

    elif model_type == DistillationModels.ConsistencyModel:

        model = create_cmd_model(image_size=args.image_size,
                                 num_channels=args.num_channels,
                                 num_res_blocks=args.num_res_blocks,
                                 channel_mult=args.channel_mult,
                                 learn_sigma=args.learn_sigma,
                                 class_cond=False,
                                 use_checkpoint=False,
                                 attention_resolutions=args.attention_resolutions,
                                 num_heads=args.num_heads,
                                 num_head_channels=-args.num_head_channels,
                                 num_heads_upsample=-args.num_heads_upsample,
                                 use_scale_shift_norm=args.use_scale_shift_norm,
                                 dropout=args.dropout,
                                 resblock_updown=args.resblock_updown,
                                 use_fp16=args.use_fp16,
                                 use_new_attention_order=args.use_new_attention_order)

        if args.dataset == Datasets.Cifar10:
            with open_url(args.teacher_model_path) as f:
                teacher_model = pickle.load(f)['ema']
        else:
            raise NotImplementedError(f'No pre-trained teacher for {args.dataset} dataset')

        teacher_diffusion = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            distillation=False,
            weight_schedule=args.weight_schedule,
        )

        target_model = copy.deepcopy(model)

        ema_scale_fn = create_ema_and_scales_fn(
            target_ema_mode=args.target_ema_mode,
            start_ema=args.start_ema,
            scale_mode=args.scale_mode,
            start_scales=args.start_scales,
            end_scales=args.end_scales,
            total_steps=args.total_training_steps,
            distill_steps_per_iter=args.distill_steps_per_iter,
        )

        cm = ConsistencyModel(model=model,
                              target_model=target_model,
                              teacher_model=teacher_model,
                              teacher_diffusion=teacher_diffusion,
                              ema_scale_function=ema_scale_fn,
                              weight_schedule=args.weight_schedule,
                              sigma_max=args.sigma_max,
                              sigma_min=args.sigma_min,
                              rho=args.rho,
                              sigma_data=args.sigma_data,
                              )

        return cm
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")


def loading_pre_trained_weights(model):
    pass


def create_koopman_model(args):
    if args.dataset == Datasets.Checkerboard:
        return KoopmanDistillOneStep(
            x0_observables_encoder=Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
            x_T_observables_encoder=Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
            x0_observables_decoder=Decoder(output_dim=args.input_dim, hidden_dim=args.hidden_dim),
            koopman_operator=torch.nn.Linear(args.hidden_dim, args.hidden_dim),
            rec_loss_type=args.rec_loss_type,
        )

    elif args.dataset == Datasets.Cifar10:
        return OneStepKoopmanCifar10(img_resolution=32, rec_loss_type=args.rec_loss_type,
                                     out_channels=args.out_channels,
                                     noisy_latent=args.noisy_latent,
                                     )

    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")


def create_cmd_model(image_size,
                     num_channels,
                     num_res_blocks,
                     channel_mult="",
                     learn_sigma=False,
                     class_cond=False,
                     use_checkpoint=False,
                     attention_resolutions="16",
                     num_heads=1,
                     num_head_channels=-1,
                     num_heads_upsample=-1,
                     use_scale_shift_norm=False,
                     dropout=0,
                     resblock_updown=False,
                     use_fp16=False,
                     use_new_attention_order=False):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 2, 2)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return SongUNet(
        img_resolution=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        dropout=dropout,
        channel_mult=channel_mult,
    )
