import argparse


def get_configs():
    """
    Load the configs, override default configs with argument parsed configs
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', type=str, required=True)

    # --- Koopman configs --- #
    if 'koopman/cifar_uncond' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond import load_arguments
        load_arguments(parser)

    elif 'koopman/cifar_uncond_precond_2_4' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond_precond_2_4 import load_arguments
        load_arguments(parser)

    elif 'koopman/cifar_uncond_adverserial_2_4' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond_adverserial_2_4 import load_arguments
        load_arguments(parser)

    elif 'koopman/cifar_uncond_dmdkoopman' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond_dmdkoopman import load_arguments
        load_arguments(parser)

    elif 'koopman/cifar_uncond_dmdkoopman_batch' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond_dmdkoopman_batch import load_arguments
        load_arguments(parser)

    elif 'koopman/cifar_uncond_vae' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond_vae import load_arguments
        load_arguments(parser)

    elif 'koopman/checkerboard_uncond' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.checkerboard_uncond import load_arguments
        load_arguments(parser)

    elif 'koopman/checkerboard_uncond_dmdkoopman' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.checkerboard_uncond_dmdkoopman import load_arguments
        load_arguments(parser)

    # --- CM configs --- #
    elif 'cm/cifar_uncond' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.cm.cifar_uncond import load_arguments
        load_arguments(parser)

    else:
        raise ModuleNotFoundError("No such config file")

    args = parser.parse_args()
    print(args)

    return args
