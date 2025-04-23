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

    elif 'koopman/cifar_uncond_fast' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_uncond_fast import load_arguments
        load_arguments(parser)

    elif 'koopman/cifar_cond' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.cifar_cond import load_arguments
        load_arguments(parser)

    elif 'koopman/checkerboard_uncond' == parser.parse_known_args()[0].config_name:
        from koopman_distillation.configs.koopman.checkerboard_uncond import load_arguments
        load_arguments(parser)
    else:
        raise ModuleNotFoundError(f"No such config file {parser.parse_known_args()[0].config_name}")

    args = parser.parse_args()
    print(args)

    return args
