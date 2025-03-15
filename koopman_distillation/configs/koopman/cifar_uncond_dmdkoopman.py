from koopman_distillation.utils.names import DistillationModels, Datasets, RecLossType


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="cifar_uncond_dmdkoopman", help='The experiment name')
    parser.add_argument('--neptune', type=bool, default=False)
    parser.add_argument('--neptune_projects', type=str, default='azencot-group/koopman-dis')
    parser.add_argument('--tags', type=list[str], default=['koopman_dmd'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="/home/bermann/functional_mapping/koopman_distillation/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Cifar10FastOneStepLoading)
    # fast loading require the path to the npy file
    parser.add_argument('--datapath', type=str,
                        # default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond')
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_dataset.npy')
    parser.add_argument('--datapath_test', type=str, default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond_test_data') # work only on non normalized data
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=602)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=1)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.KoopmanDistillOneStepDMD)
    parser.add_argument('--ema_rate', type=list[float], default=[0.999, 0.9999, 0.9999432189950708])
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--nonlinear_koopman', type=bool, default=False)

    # losses
    parser.add_argument('--rec_loss_type', type=str, default=RecLossType.LPIPS)

    # --- sampling --- #
    parser.add_argument('--data_shape', type=list[int], default=(3, 32, 32))

    # --- koopman parameters --- #
    parser.add_argument('--noisy_latent', type=float, default=0.4)
