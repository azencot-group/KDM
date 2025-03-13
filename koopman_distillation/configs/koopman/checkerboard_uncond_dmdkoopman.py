from koopman_distillation.utils.names import DistillationModels, Datasets, RecLossType


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="check_check", help='The experiment name')
    parser.add_argument('--neptune', type=bool, default=False)
    parser.add_argument('--neptune_projects', type=str, default='azencot-group/koopman-dis')
    parser.add_argument('--tags', type=list[str], default=['checkerboard'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="/home/bermann/functional_mapping/koopman_distillation/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Checkerboard)
    parser.add_argument('--datapath', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/checkerboard/sol.npy')
    parser.add_argument('--datapath_test', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=6)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=201)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=2)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.KoopmanDistillOneStepDMD)
    parser.add_argument('--ema_rate', type=list[float], default=[])

    # --- sampling --- #
    parser.add_argument('--data_shape', type=tuple[int], default=(2,))

    # --- koopman parameters --- #
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=2)

    # losses
    parser.add_argument('--rec_loss_type', type=str, default=RecLossType.L2)
