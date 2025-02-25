import torch.nn

from koopman_distillation.model.koopman_distillator import KoopmanDistillOneStep
from koopman_distillation.model.modules.checkerboard import Encoder, Decoder
from koopman_distillation.model.modules.cifar10 import OneStepKoopmanCifar10
from koopman_distillation.other_methods.consistency_models.models.unet import UNetModel
from koopman_distillation.utils.names import DistillationModels, Datasets


def create_distillation_model(model_type: DistillationModels, args):
    if model_type == DistillationModels.OneStepKOD:
        return create_koopman_model(args)
    
    elif model_type == DistillationModels.ConsistencyModels:
        return create_cmd_model(image_size=args.image_size,
                                num_channels=args.num_channels,
                                num_res_blocks=args.num_res_blocks,
                                channel_mult=args.channel_mult,
                                learn_sigma=args.learn_sigma,
                                class_cond=args.class_cond,
                                use_checkpoint=args.use_checkpoint,
                                attention_resolutions=args.attention_resolutions,
                                num_heads=args.num_heads,
                                num_head_channels=-args.num_head_channels,
                                num_heads_upsample=-args.num_heads_upsample,
                                use_scale_shift_norm=args.use_scale_shift_norm,
                                dropout=args.dropout,
                                resblock_updown=args.resblock_updown,
                                use_fp16=args.use_fp16,
                                use_new_attention_order=args.use_new_attention_order)
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
            koopman_operator=torch.nn.Linear(args.hidden_dim, args.hidden_dim))

    elif args.dataset == Datasets.Cifar10:
        return OneStepKoopmanCifar10(img_resolution=32)

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
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
    )
