class Datasets:
    Cifar10 = "Cifar10"
    Cifar10FastOneStepLoading = "Cifar10FastOneStepLoading"
    Checkerboard = "Checkerboard"


class DistillationModels:
    OneStepKOD = "Koopman Operator Distillation"
    KoopmanDistillOneStepDMD = "Koopman Operator Distillation DMD"
    OneStepKoopmanCifar10DMDPredictMatrix = "Koopman Operator Distillation DMD Matrix Producer"
    ConsistencyModel = "Consistency Distillation Model"


# --- models configurations --- #
class RecLossType:
    L2 = "L2"
    LPIPS = "LPIPS"
    BOTH = "BOTH"


