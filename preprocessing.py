
import numpy as np

from enumerations import ClassificationFormat
from framework_component import Component


class BasePreProcessing(Component):
    def __init__(self):
        pass

    def fit_numerical(self, column_name:str, values:np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def transform_numerical(self, column_name:str, values: np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def fit_categorical(self, column_name:str, values:np.array):
        raise NotImplementedError("Please override this base class with a custom implementation")

    def transform_categorical(self, column_name:str, values:np.array, expected_categorical_format:ClassificationFormat):
        raise NotImplementedError("Please override this base class with a custom implementation")
