from utils.names import DistillationModels, Datasets, RecLossType, CondType, \
    EigenSpecKoopmanLossTypes


def load_arguments(parser) -> None:
    # --- general --- #
    parser.add_argument('--experiment_name', type=str, default="afhq_uncond", help='The experiment name')
    parser.add_argument('--neptune', type=bool, default=False)
    parser.add_argument('--neptune_projects', type=str, default='<your_hub>/koopman-dis')
    parser.add_argument('--tags', type=str, nargs='+', default=['afhq'])

    # --- artifacts --- #
    parser.add_argument('--output_prefix_path', type=str,
                        default="<your_output_path>/results")

    # --- data --- #
    parser.add_argument('--dataset', type=str, default=Datasets.AFHQ_250K)
    # fast loading require the path to the npy file
    parser.add_argument('--datapath', type=str, default='<your_data_path>')
    parser.add_argument('--datapath_test', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=6)
    parser.add_argument('--image_resolution', type=int, default=64)

    # --- training --- #
    parser.add_argument('--iterations', type=int, default=800001)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--print_every', type=float, default=200)

    # --- models --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.OneStepKOD)
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 2, 2])
    parser.add_argument('--out_channels', type=int, default=1)
    parser.add_argument('--model_channels', type=int, default=32)

    # losses
    parser.add_argument('--rec_loss_type', type=str, default=RecLossType.LPIPS)
    parser.add_argument('--psudo_huber_c', type=float, default=0.03)
    parser.add_argument('--w_latent', type=float, default=1)
    parser.add_argument('--w_rec', type=float, default=1)
    parser.add_argument('--w_push', type=float, default=1)

    # --- sampling --- #
    parser.add_argument('--data_shape', type=list[int], default=(3, 64, 64))

    # --- koopman parameters --- #
    parser.add_argument('--noisy_latent', type=float, default=0.4)
    parser.add_argument('--initial_noise_factor', type=float, default=80)
    parser.add_argument('--add_sampling_noise', type=float, default=0.4)
    parser.add_argument('--cond_type', type=str, default=CondType.Uncond)
    parser.add_argument('--koopman_loss_type', type=str, default=EigenSpecKoopmanLossTypes.NoLoss)
    parser.add_argument('--label_dim', type=int, default=0)
    # enable using linear projection to define specific latent space size, if none no linear projections will be added
    parser.add_argument('--linear_proj', type=int, default=512)

    # --- adversarial --- #
    parser.add_argument('--advers', type=bool, default=True)
    parser.add_argument('--advers_w', type=float, default=1)
