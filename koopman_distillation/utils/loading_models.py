import torch.nn

from koopman_distillation.models.koopman_model_checkerboard import Encoder, Decoder, KoopmanDistillOneStep
from koopman_distillation.models.koopman_model_cifar10 import AdversarialOneStepKoopmanCifar10, \
    AdversarialOneStepKoopmanCifar10Fast
from koopman_distillation.utils.names import DistillationModels, Datasets


def create_distillation_model(model_type: DistillationModels, args):
    if model_type == DistillationModels.OneStepKOD or model_type == DistillationModels.FastOneStepKOD:
        return create_koopman_model(args)
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")


def loading_pre_trained_weights(model):
    pass


def create_koopman_model(args):
    if args.dataset == Datasets.Checkerboard:
        # todo - implement adversarial option to the loss of this models
        return KoopmanDistillOneStep(
            x0_observables_encoder=Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
            x_T_observables_encoder=Encoder(input_dim=args.input_dim, hidden_dim=args.hidden_dim),
            x0_observables_decoder=Decoder(output_dim=args.input_dim, hidden_dim=args.hidden_dim),
            koopman_operator=torch.nn.Linear(args.hidden_dim, args.hidden_dim),
            rec_loss_type=args.rec_loss_type,
        )

    elif args.dataset in [Datasets.Cifar10_1M_Uncond, Datasets.Cifar10_1M_Cond]:
        if DistillationModels.OneStepKOD == args.distillation_model:
            return AdversarialOneStepKoopmanCifar10(img_resolution=32,
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
                                                    noisy_latent_after_push=args.noisy_latent_after_push,
                                                    contrastive_estimation=args.contrastive_estimation,
                                                    contrast_x0_zT=args.contrast_x0_zT,
                                                    contrast_x0_z0=args.contrast_x0_z0,
                                                    contrast_xT_zT=args.contrast_xT_zT,
                                                    gnn_regularization=args.gnn_regularization,
                                                    advers_w=args.advers_w,
                                                    w_latent=args.w_latent,
                                                    w_rec=args.w_rec,
                                                    w_push=args.w_push,
                                                    koopman_loss_type=args.koopman_loss_type,
                                                    )
        elif DistillationModels.FastOneStepKOD == args.distillation_model:
            return AdversarialOneStepKoopmanCifar10Fast(img_resolution=32,
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
                                                        noisy_latent_after_push=args.noisy_latent_after_push,
                                                        contrastive_estimation=args.contrastive_estimation,
                                                        contrast_x0_zT=args.contrast_x0_zT,
                                                        contrast_x0_z0=args.contrast_x0_z0,
                                                        contrast_xT_zT=args.contrast_xT_zT,
                                                        gnn_regularization=args.gnn_regularization,
                                                        advers_w=args.advers_w,
                                                        w_latent=args.w_latent,
                                                        w_rec=args.w_rec,
                                                        w_push=args.w_push,
                                                        koopman_loss_type=args.koopman_loss_type,
                                                        )
        else:
            raise NotImplementedError(f"Dataset {args.dataset} not implemented")
