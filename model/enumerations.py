
from enum import Enum

class ClassificationFormat(Enum):
    """
    The form of the input tensor for the model
    """
    Integers = 0,
    OneHot = 1


class EvaluationDatasetSampling(Enum):
    """
    How to choose evaluation samples from the raw dataset
    """
    LastRows = 0

    RandomRows  = 1

    FilterColumn = 2

