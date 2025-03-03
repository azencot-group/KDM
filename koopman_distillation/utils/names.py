class Datasets:
    Cifar10 = "Cifar10"
    Checkerboard = "Checkerboard"


class DistillationModels:
    OneStepKOD = "Koopman Operator Distillation"
    ConsistencyModel = "Consistency Distillation Model"


# --- models configurations --- #
class RecLossType:
    L2 = "L2"
    LPIPS = "LPIPS"
    BOTH = "BOTH"
