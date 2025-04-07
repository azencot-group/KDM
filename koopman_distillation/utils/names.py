class Datasets:
    Cifar10_1M_Uncond = "Cifar10 1 million unconditional"
    Cifar10FastOneStepLoading = "Cifar10FastOneStepLoading"
    Checkerboard = "Checkerboard"


class DistillationModels:
    OneStepKOD = "Koopman Operator Distillation"


# --- models configurations --- #
class RecLossType:
    L2 = "L2"
    LPIPS = "LPIPS"
    BOTH = "BOTH"
    Huber = "Huber"
    Wess = "Wess"
