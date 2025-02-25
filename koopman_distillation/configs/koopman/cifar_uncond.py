from koopman_distillation.utils.names import DistillationModels, Datasets


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="cifar_uncond", help='The experiment name')
    parser.add_argument('--neptune', type=bool, default=False)
    parser.add_argument('--neptune_projects', type=str, default='azencot-group/koopman-dis')
    parser.add_argument('--tags', type=list[str], default=['v0', 'also_perceptual_eval_off'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="/home/bermann/functional_mapping/koopman_distillation/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.Cifar10)
    parser.add_argument('--datapath', type=str,
                        default='/cs/cs_groups/azencot_group/functional_diffusion/data_for_distillation/cifar32uncond')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=6)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=250)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--print_every', type=float, default=1)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.OneStepKOD)

    # losses
    parser.add_argument('--rec_loss_type', type=list[str], default=['l2', 'lpips', 'mixed'])

    # --- sampling --- #
    parser.add_argument('--data_shape', type=list[int], default=(3, 32, 32))

    # --- koopman parameters --- #
