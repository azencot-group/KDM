class Datasets:
    Cifar10_1M_Uncond = "Cifar10 1 million unconditional"
    Cifar10_1M_Cond = "Cifar10 1 million conditional"
    Checkerboard = "Checkerboard"


class DistillationModels:
    OneStepKOD = "Koopman Operator Distillation"
    FastOneStepKOD = "Koopman Operator Distillation Fast"


# --- models configurations --- #
class RecLossType:
    L2 = "L2"
    LPIPS = "LPIPS"
    BOTH = "BOTH"
    Huber = "Huber"
    Wess = "Wess"

class CondType:
    Uncond = "Uncond"
    OnlyEncDec = "OnlyEncDec"
    KoopmanMatrixAddition = "KoopmanMatrixAddition"


class EigenSpecKoopmanLossTypes:
    NoLoss = "NoLoss"
    Uniform = "Uniform"
    GradualUniform = "GradualUniform"
    OnTheCircle = "OnTheCircle"