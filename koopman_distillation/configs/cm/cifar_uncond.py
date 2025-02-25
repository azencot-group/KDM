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
    parser.add_argument('--lr', type=float, default=0.000008)
    parser.add_argument('--print_every', type=float, default=1)

    # --- model --- #
    parser.add_argument('--distillation_model', type=str, default=DistillationModels.OneStepKOD)

    # --- sampling --- #
    parser.add_argument('--data_shape', type=list[int], default=(3, 32, 32))

    # --- cm parameters --- #
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--num_channels', type=int, default=192)
    parser.add_argument('--num_res_blocks', type=int, default=3)
    parser.add_argument('--channel_mult', type=str, default="")
    parser.add_argument('--learn_sigma', type=bool, default=False)
    parser.add_argument('--attention_resolutions', type=list[int], default=[32,16,8])
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_head_channels', type=int, default=64)
    parser.add_argument('--num_heads_upsample', type=int, default=-1)
    parser.add_argument('--use_scale_shift_norm', type=bool, default=True)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--resblock_updown', type=bool, default=True)
    parser.add_argument('--use_fp16', type=bool, default=True)
    parser.add_argument('--use_new_attention_order', type=bool, default=False)


# --training_mode consistency_distillation --target_ema_mode fixed --start_ema 0.95
# --scale_mode fixed --start_scales 40 --schedule_sampler uniform --weight_schedule uniform
