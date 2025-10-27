

import numpy as np
import tensorflow as tf

from classification_head import BaseClassificationHead

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras.layers import  Lambda, GlobalAveragePooling1D



class GlobalAveragePoolingClassificationHead(BaseClassificationHead):
    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""
        return GlobalAveragePooling1D(name=f"{prefix}global_avg_pooling_1d")(X)

    @property
    def name(self) -> str:
        return "Global Average Pooling"

    @property
    def parameters(self) -> dict:
        return {}


class LastTokenClassificationHead(BaseClassificationHead):
    def __init__(self):
        super().__init__()

    def apply(self, X, prefix: str = None):
        if prefix is None:
            prefix = ""

        x = Lambda(lambda x: x[..., -1, :], name=f"{prefix}slice_last")(X)

        return x

    @property
    def name(self) -> str:
        return "Last Token"

    @property
    def parameters(self) -> dict:
        return {}


