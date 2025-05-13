class Datasets:
    Cifar10_1M_Uncond = "Cifar10 1 million unconditional"
    Cifar10_1M_Uncond_FM = "Cifar10 1 million unconditional Flow Matching"
    Cifar10_1M_Cond = "Cifar10 1 million conditional"
    FFHQ_1M = "FFHQ_1M"
    AFHQ_250K = "AFHQ_250K"
    Checkerboard = "Checkerboard"


class DistillationModels:
    OneStepKOD = "Koopman Operator Distillation"
    DecomposedOneStepKOD = "Koopman Operator Distillation With Decomposed Matrix"


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
