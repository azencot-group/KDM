import torch.nn

from koopman_distillation.models.koopman_model_checkerboard import Encoder, Decoder, KoopmanDistillModelCheckerBoard
from koopman_distillation.models.koopman_model import AdversarialOneStepKoopmanCifar10, \
    AdversarialOneStepKoopmanCifar10Decomposed
from koopman_distillation.utils.names import DistillationModels, Datasets


def create_distillation_model(model_type: DistillationModels, args):
    if model_type == DistillationModels.OneStepKOD or model_type == DistillationModels.DecomposedOneStepKOD:
        return create_koopman_model(args)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")


def create_koopman_model(args):
    if args.dataset == Datasets.Checkerboard:
        return KoopmanDistillModelCheckerBoard(
            x0_observables_encoder=Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
            x_T_observables_encoder=Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
            x0_observables_decoder=Decoder(output_dim=args.input_dim, hidden_dim=args.hidden_dim),
            koopman_operator=torch.nn.Linear(args.hidden_dim, args.hidden_dim),
            rec_loss_type=args.rec_loss_type,
        )

    elif args.dataset in [Datasets.Cifar10_1M_Uncond, Datasets.Cifar10_1M_Cond, Datasets.FFHQ_1M, Datasets.AFHQ_250K,
                          Datasets.Cifar10_1M_Uncond_FM]:
        if DistillationModels.OneStepKOD == args.distillation_model:
            return AdversarialOneStepKoopmanCifar10(img_resolution=args.image_resolution,
                                                    rec_loss_type=args.rec_loss_type,
                                                    out_channels=args.out_channels,
                                                    noisy_latent=args.noisy_latent,
                                                    add_sampling_noise=args.add_sampling_noise,
                                                    model_channels=args.model_channels,
                                                    channel_mult=args.channel_mult,
                                                    psudo_huber_c=args.psudo_huber_c,
                                                    initial_noise_factor=args.initial_noise_factor,
                                                    cond_type=args.cond_type,
                                                    label_dim=args.label_dim,
                                                    advers_w=args.advers_w,
                                                    w_latent=args.w_latent,
                                                    w_rec=args.w_rec,
                                                    w_push=args.w_push,
                                                    koopman_loss_type=args.koopman_loss_type,
                                                    linear_proj=args.linear_proj,
                                                    )
        elif DistillationModels.DecomposedOneStepKOD == args.distillation_model:
            return AdversarialOneStepKoopmanCifar10Decomposed(img_resolution=args.image_resolution,
                                                              rec_loss_type=args.rec_loss_type,
                                                              out_channels=args.out_channels,
                                                              noisy_latent=args.noisy_latent,
                                                              add_sampling_noise=args.add_sampling_noise,
                                                              model_channels=args.model_channels,
                                                              channel_mult=args.channel_mult,
                                                              psudo_huber_c=args.psudo_huber_c,
                                                              initial_noise_factor=args.initial_noise_factor,
                                                              cond_type=args.cond_type,
                                                              label_dim=args.label_dim,
                                                              advers_w=args.advers_w,
                                                              w_latent=args.w_latent,
                                                              w_rec=args.w_rec,
                                                              w_push=args.w_push,
                                                              koopman_loss_type=args.koopman_loss_type,
                                                              linear_proj=args.linear_proj,
                                                              )
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")
