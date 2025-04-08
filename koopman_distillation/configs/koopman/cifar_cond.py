from koopman_distillation.utils.names import DistillationModels, Datasets, RecLossType, CondType


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="cifar_uncond", help='The experiment name')
    parser.add_argument("--neptune", action='store_true', help="log to neptune")
    parser.add_argument('--neptune_projects', type=str, default='azencot-group/koopman-dis')
    parser.add_argument('--tags', type=list[str], default=['Adversarial', 'cond', 'Sanity Check new_main_7_4'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="/cs/azencot_fsas/functional_diffusion/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Cifar10_1M_Cond)
    # fast loading require the path to the npy file
    parser.add_argument('--datapath', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32cond_test_1M')
    parser.add_argument('--datapath_test', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_test_data')  # work only on non normalized data
    parser.add_argument('--batch_size', type=int, default=384)
    parser.add_argument('--num_workers', type=int, default=6)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=-1, help='number of iterations is set in the main')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=75)
    parser.add_argument("--fp16", action='store_true', help="use mixed precision")
    parser.add_argument('--seed', type=int, default=42)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.OneStepKOD)
    parser.add_argument('--ema_rate', type=list[float], default=[0.999, 0.9999, 0.9999432189950708])
    parser.add_argument('--channel_mult', type=list[int], default=[1, 2, 2, 2])
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--model_channels', type=int, default=64)
    parser.add_argument('--nonlinear_koopman', type=bool, default=False)

    # losses
    parser.add_argument('--rec_loss_type', type=str, default=RecLossType.LPIPS)
    parser.add_argument('--mixup', type=float, default=0)
    parser.add_argument('--psudo_huber_c', type=float, default=0.03)

    # --- sampling --- #
    parser.add_argument('--data_shape', type=list[int], default=(3, 32, 32))

    # --- koopman parameters --- #
    parser.add_argument('--noisy_latent', type=float, default=0.4)
    parser.add_argument('--initial_noise_factor', type=float, default=80)
    parser.add_argument('--add_sampling_noise', type=float, default=0.4)
    parser.add_argument('--cond_type', type=str, default=CondType.KoopmanMatrixAddition)
    parser.add_argument('--label_dim', type=int, default=10)

    # --- adversarial --- #
    parser.add_argument('--advers', type=bool, default=True)

    # --- multi gpu --- #
    parser.add_argument('--gpu_num', type=int, default=3)
    parser.add_argument('--nar', type=int, default=2, help='num_accumulation_rounds')


