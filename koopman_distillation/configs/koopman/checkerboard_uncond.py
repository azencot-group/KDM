from koopman_distillation.utils.names import DistillationModels, Datasets


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="check_check", help='The experiment name')

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="/home/bermann/functional_mapping/koopman_distillation/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Checkerboard)
    parser.add_argument('--datapath', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/checkerboard/sol.npy')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=6)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=1001)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=50)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.OneStepKOD)

    # --- sampling --- #
    parser.add_argument('--iterations', type=tuple[int], default=(2,))

    # --- koopman parameters --- #
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=2)
