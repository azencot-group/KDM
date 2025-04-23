from koopman_distillation.utils.names import DistillationModels, Datasets, RecLossType, CondType, \
    EigenSpecKoopmanLossTypes


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="cifar_uncond", help='The experiment name')
    parser.add_argument('--neptune', type=bool, default=False)
    parser.add_argument('--neptune_projects', type=str, default='azencot-group/koopman-dis')
    parser.add_argument('--tags', type=str, nargs='+', default=['bigger_latent_space'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="/home/bermann/functional_mapping/koopman_distillation/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Cifar10_1M_Uncond)
    # fast loading require the path to the npy file
    parser.add_argument('--datapath', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_1M')
    parser.add_argument('--datapath_test', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_test_data')  # work only on non normalized data
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--dataset_subset', type=str, default='no_subset', choices=['50k', '250k', '500k', 'no_subset'])

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=800001)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=200)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.FastOneStepKOD)
    parser.add_argument('--ema_rate', type=list[float], default=[0.999, 0.9999, 0.9999432189950708])
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--model_channels', type=int, default=64)
    parser.add_argument('--nonlinear_koopman', type=bool, default=False)

    # losses
    parser.add_argument('--rec_loss_type', type=str, default=RecLossType.LPIPS)
    parser.add_argument('--psudo_huber_c', type=float, default=0.03)

    parser.add_argument('--w_latent', type=float, default=1)
    parser.add_argument('--w_rec', type=float, default=1)
    parser.add_argument('--w_push', type=float, default=1)

    # --- sampling --- #
    parser.add_argument('--data_shape', type=list[int], default=(3, 32, 32))

    # --- koopman parameters --- #
    parser.add_argument('--noisy_latent', type=float, default=0.4)
    parser.add_argument('--noisy_latent_after_push', type=float, default=0)
    parser.add_argument('--initial_noise_factor', type=float, default=80)
    parser.add_argument('--add_sampling_noise', type=float, default=0.4)
    parser.add_argument('--cond_type', type=str, default=CondType.Uncond)
    parser.add_argument('--koopman_loss_type', type=str, default=EigenSpecKoopmanLossTypes.Uniform)
    parser.add_argument('--label_dim', type=int, default=0)
    parser.add_argument('--contrastive_estimation', type=float, default=0)
    parser.add_argument('--contrast_x0_zT', type=float, default=0)
    parser.add_argument('--contrast_x0_z0', type=float, default=0)
    parser.add_argument('--contrast_xT_zT', type=float, default=0)


    # --- adversarial --- #
    parser.add_argument('--advers', type=bool, default=True)
    parser.add_argument('--advers_w', type=float, default=1)

    # --- gnn --- #
    parser.add_argument('--gnn_regularization', type=float, default=0)