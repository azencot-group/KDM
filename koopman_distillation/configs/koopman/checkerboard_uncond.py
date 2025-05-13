from koopman_distillation.utils.names import DistillationModels, Datasets, RecLossType, CondType


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="checker", help='The experiment name')
    parser.add_argument('--neptune', type=bool, default=False)
    parser.add_argument('--neptune_projects', type=str, default='project_name')
    parser.add_argument('--tags', type=list[str], default=['checkerboard'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str, default="./results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Checkerboard)
    parser.add_argument('--datapath', type=str, default='./data/checkerboard/sampled_dataset.npy')
    parser.add_argument('--datapath_test', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=6)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=20001)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=200)

    # --- models --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.OneStepKOD)
    parser.add_argument('--cond_type', type=str, default=CondType.Uncond)

    # --- sampling --- #
    parser.add_argument('--data_shape', type=tuple[int], default=(2,))

    # --- koopman parameters --- #
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--input_dim', type=int, default=2)

    # losses
    parser.add_argument('--rec_loss_type', type=str, default=RecLossType.L2)

    # adversarial
    parser.add_argument('--advers', type=bool, default=False)  # not implemented for this model
    parser.add_argument('--advers_w', type=float, default=0)  # not implemented for this model
