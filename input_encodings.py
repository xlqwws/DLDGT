

import warnings
from enum import Enum
from typing import List

from input_encoding import BaseInputEncoding
from enumerations import ClassificationFormat

try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras

from keras.layers import Dense, Concatenate,  Lambda
import tensorflow as tf

class NoInputEncoder(BaseInputEncoding):
    def apply(self, X, prefix:str=None):

        numerical_feature_inputs = X[:self.model_input_specification.n_numeric_features]
        categorical_feature_inputs = X[self.model_input_specification.n_numeric_features:]

        if self.model_input_specification.categorical_format == ClassificationFormat.Integers:
            warnings.warn("It doesn't make sense to be using integer based inputs without encoding!")
            categorical_feature_inputs = [Lambda(lambda x: tf.cast(x, tf.float32))(c) for c in categorical_feature_inputs]

        concat = Concatenate()(numerical_feature_inputs + categorical_feature_inputs)

        return concat

    @property
    def name(self):
        return "No Input Encoding"

    @property
    def parameters(self):
        return {}

    @property
    def required_input_format(self) -> ClassificationFormat:
        return ClassificationFormat.OneHot


class RecordLevelEmbed(BaseInputEncoding):
    def __init__(self, embed_dimension: int, project:bool = False):
        super().__init__()

        self.embed_dimension: int = embed_dimension
        self.project: bool = project

    @property
    def name(self):
        if self.project:
            return "Record Level Projection"
        return "Record Level Embedding"

    @property
    def parameters(self):
        return {
            "dimensions_per_feature": self.embed_dimension
        }

    def apply(self, X:List[keras.Input], prefix: str = None):
        if prefix is None:
            prefix = ""

        assert self.model_input_specification.categorical_format == ClassificationFormat.OneHot

        x = Concatenate(name=f"{prefix}feature_concat", axis=-1)(X)
        x = Dense(self.embed_dimension, activation="linear", use_bias=not self.project, name=f"{prefix}embed")(x)

        return x

    @property
    def required_input_format(self) -> ClassificationFormat:
        return ClassificationFormat.OneHot


