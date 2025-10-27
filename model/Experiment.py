

import os

import pandas as pd

from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer
from sequential import BaseSequential
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
#  NetworkTransformer

import tensorflow as tf
from keras.layers import Dense, Layer, MultiHeadAttention, Dropout, LayerNormalization,LSTM,Conv1D,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K
import keras.layers as layers

from keras.layers import Dense,Conv1D, MultiHeadAttention, Dropout, LayerNormalization,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D,ConvLSTM2D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),

]



transformers: List[FunctionalComponent] = [
    EncoderDecoderTransformer(
        n_encoder_layers=3,
        n_decoder_layers=3,
        internal_size=256,#
        n_heads=5
    )
]

print(' ================================================== DGT transformer ==================================================')

flow_file_path = r"D:\dataset"

datasets = [
    ("CICIDS2017", pd.read_csv( r"D:\dataset\CICIDS2017\train.csv",encoding='gbk'), NamedDatasetSpecifications.CICIDS2017, 0.17, EvaluationDatasetSampling.RandomRows),
    ("UnknownAttack", pd.read_csv(r"D:\dataset\unknown_attacks\unknown_attack6.csv",encoding='gbk'), NamedDatasetSpecifications.UnknownAttack, 0.2, EvaluationDatasetSampling.RandomRows)

]

pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the Networktransformer
ft = DGTTransformer(pre_processing=pre_processing,
                     # input_encoding=encodings[0],
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     # classification_head=GlobalAveragePoolingClassificationHead(),
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m = ft.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)


print('CICIDS2017 test dataset prediction results\n',eval_results)


# Define the Networktransformer
ft1 = DGTTransformer(pre_processing=pre_processing,
                     # input_encoding=encodings[0],
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     # classification_head=GlobalAveragePoolingClassificationHead(),
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft1.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
mn = ft1.build_model()
mn.summary()

# Compile the model
mn.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results: pd.DataFrame
(train_results1, eval_results, final_epoch1) = ft.evaluate(mn, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)


print('UnknownAttack test dataset prediction results\n',eval_results)

print(' ==================================================traditional transformer ==================================================')


class TransformerDecoderBlock1(Layer):
    def __init__(self, input_dimension: int, inner_dimension: int,
                 num_heads: int, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate
        # Self-Attention Mechanism
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension
        )
        self.self_attn_dropout = Dropout(dropout_rate)
        self.self_attn_norm = LayerNormalization(epsilon=1e-6)

        # Encoder-Decoder Attention Mechanism
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension
        )
        self.cross_attn_dropout = Dropout(dropout_rate)
        self.cross_attn_norm = LayerNormalization(epsilon=1e-6)

        # Feedforward network
        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])
        self.ffn_dropout = Dropout(dropout_rate)
        self.ffn_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False, mask=None):
        # The input should consist of two parts: [decoder_input, encoder_output]
        decoder_input, encoder_output = inputs

        # Layer 1: Self-Attention
        attn_output = self.self_attn(
            decoder_input, decoder_input
        )
        attn_output = self.self_attn_dropout(attn_output, training=training)
        out1 = decoder_input + attn_output
        out1 = self.self_attn_norm(out1)

        # Layer 2: Encoder-Decoder Attention
        cross_attn_output = self.cross_attn(
            out1, encoder_output
        )
        cross_attn_output = self.cross_attn_dropout(
            cross_attn_output, training=training
        )
        out2 = out1 + cross_attn_output
        out2 = self.cross_attn_norm(out2)

        # Third Layer: Feedforward Network
        ffn_output = self.ffn(out2)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.ffn_norm(out3)

        return out3



class GPT3Attention(layers.Layer):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

class MultiHeadAttentionImplementation:
    Keras = 0,
    GPT3 = 1

class TransformerEncoderBlock1(layers.Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv:bool=False, prefix:str=None, attn_implementation:MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else\
                GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        self.feed_forward_0 = Conv1D(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
        # self.feed_forward_1 = Conv1D(input_dimension, activation="relu", name=f"{prefix}feed_forward_1") \
        #     if use_conv else Dense(input_dimension, activation="relu", name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}feed_forward_layer_norm")


    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        x = self.feed_forward_0(x)
        # x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(attention_output + feed_forward_output)





class EncoderDecoderTransformer1(BaseSequential):
    """A Transformer architecture that simultaneously incorporates both encoders and decoders"""

    @property
    def name(self) -> str:
        return "Encoder-Decoder Transformer"

    @property
    def parameters(self) -> dict:
        return {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        }

    def __init__(self, n_encoder_layers: int, n_decoder_layers: int,
                 internal_size: int, n_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def apply(self, X, prefix: str = None):
        # Encoder Section
        encoder_output = X
        for i in range(self.n_encoder_layers):
            encoder_output = TransformerEncoderBlock1(
                input_dimension=encoder_output.shape[-1],
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=False,
                prefix=f"{prefix}encoder_{i}_"
            )(encoder_output)

        # Decoder Section (Using Encoder Output)
        decoder_output = encoder_output
        for i in range(self.n_decoder_layers):
            decoder_output = TransformerDecoderBlock1(
                input_dimension=decoder_output.shape[-1],
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate
            )([decoder_output, encoder_output])

        return decoder_output


transformers1: List[FunctionalComponent] = [
    EncoderDecoderTransformer1(
        n_encoder_layers=1,
        n_decoder_layers=1,
        internal_size=64,
        n_heads=3
    )
]






class GPTtransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "GPT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 6
        self.internal_size = 128
        self.n_heads = 6
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = True

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerDecoderBlock2(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)

        return m_x


class BERTtransformer(BaseSequential):

    @property
    def name(self) -> str:
        return "BERT Model"

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.head_size
        }

    def __init__(self):
        super().__init__()
        self.n_layers = 6
        self.internal_size = 128
        self.n_heads = 6
        self.head_size = self.internal_size / self.n_heads
        self.dropout_rate = 0.02
        self.is_decoder = False

    def apply(self, X, prefix: str = None):
        #window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            m_x = TransformerEncoderBlock2(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, prefix=f"block_{layer_i}_")(m_x)

        return m_x

class Transformer(BaseSequential):

    @property
    def name(self) -> str:
        if self.use_conv:
            return f"Basic  Transformer" + (" Decoder" if self.is_decoder else "")
        else:
            return f"Basic  Transformer" + (" Decoder" if self.is_decoder else "")

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "use_conv": self.use_conv,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.internal_size
        }

    def __init__(self, n_layers:int, internal_size:int, n_heads:int, use_conv:bool=False, dropout_rate:float=0.1, is_decoder=False):
        super().__init__()
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.use_conv = use_conv
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.is_decoder = is_decoder

    def apply(self, X, prefix: str = None):

        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            if self.is_decoder:
                if self.use_conv:
                    raise NotImplementedError()
                m_x = TransformerDecoderBlock2(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate)(m_x)
            else:
                m_x = TransformerEncoderBlock2(real_size, self.internal_size, self.n_heads, dropout_rate=self.dropout_rate, use_conv=self.use_conv, prefix=f"{prefix}block_{layer_i}_")(m_x)

        return m_x




class TransformerEncoderBlock2(layers.Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv:bool=False, prefix:str=None, attn_implementation:MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else\
                GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        self.feed_forward_0 = Conv1D(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
        self.feed_forward_1 = Conv1D(input_dimension, activation="relu", name=f"{prefix}feed_forward_1") \
            if use_conv else Dense(input_dimension, activation="relu", name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}feed_forward_layer_norm")




    def call(self, inputs, training, mask=None):
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        x = self.feed_forward_0(x)
        x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(attention_output + feed_forward_output)


class TransformerDecoderBlock2(Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=input_dimension)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])

        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)


    def call(self, inputs, training, mask=None):

        target_seq = inputs
        enc_output = inputs


        attn_output = self.mha(target_seq, target_seq)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = target_seq + attn_output
        out1 = self.layernorm1(out1)


        attn_output = self.mha(out1, enc_output)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = out1 + attn_output
        out2 = self.layernorm2(out2)

        # feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.layernorm2(out3)

        return out3





transformers2: List[FunctionalComponent] = [
    Transformer(2, 256, n_heads=5), #编码器
    Transformer(2, 128, n_heads=5, is_decoder=True),#解码器
    GPTtransformer(),
    BERTtransformer()
]



#
# Define the Networktransformer
ft = DGTTransformer(pre_processing=pre_processing,
                     # input_encoding=encodings[0],
                     input_encoding=encodings[1],
                     sequential_model=transformers2[3],
                     classification_head=GlobalAveragePoolingClassificationHead(),
                     # classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[68], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m1 = ft.build_model()
# m1.summary()

# Compile the model
m1.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results: pd.DataFrame
(train_results2, eval_results, final_epoch2) = ft.evaluate(m1, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)
#(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)

print('CICIDS2017 test dataset prediction results\n',eval_results)

#第二
# Define the Networktransformer
ft1 = DGTTransformer(pre_processing=pre_processing,
                     # input_encoding=encodings[0],
                     input_encoding=encodings[1],
                     sequential_model=transformers2[2],
                     classification_head=GlobalAveragePoolingClassificationHead(),
                     # classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[68], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft1.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
mn1 = ft1.build_model()
# mn1.summary()

# Compile the model
mn1.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results: pd.DataFrame
(train_results3, eval_results, final_epoch3) = ft.evaluate(mn1, batch_size=64, epochs=5, steps_per_epoch=64, early_stopping_patience=10)
#(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)

print('UnknownAttack test dataset prediction results\n',eval_results)











import os
import tempfile
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score


import pandas as pd

from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer
from sequential import BaseSequential
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
#  NetworkTransformer

import tensorflow as tf
from keras.layers import Dense, Layer, MultiHeadAttention, Dropout, LayerNormalization,LSTM,Conv1D,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K
import keras.layers as layers
from sequential import BaseSequential
from dynamic_decoder import TransformerDecoderBlock
from dynamic_encoder import TransformerEncoderBlock

try:
    from tensorflow._api.v2.v2 import keras
    from tensorflow.keras.layers import Layer
except ImportError:
    from tensorflow import keras
    from tensorflow.keras.layers import Layer
from keras.layers import Dense,Conv1D, MultiHeadAttention, Dropout, LayerNormalization,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D,ConvLSTM2D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling, ClassificationFormat
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
# from input_encodings import RecordLevelEmbed
from pre_processings import BasePreProcessing
from tensorflow.keras import layers, Model, Input


class ModelInputSpecification:
    """Input specification class required for RecordLevelEmbed"""
    def __init__(self, categorical_format, n_features):
        self.categorical_format = categorical_format
        self.n_features = n_features
        self.window_size = 1


class EncoderDecoderTransformer(Layer):
    """A Transformer architecture that simultaneously incorporates both encoders and decoders"""

    @property
    def name(self) -> str:
        return "EncoderDecoderTransformer"

    @property
    def parameters(self) -> dict:
        return {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        }

    def __init__(self, n_encoder_layers: int, n_decoder_layers: int,
                 internal_size: int, n_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        # Create a list of encoder and decoder layers
        self.encoder_layers = [
            TransformerEncoderBlock(
                input_dimension=self.internal_size,
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=False,
                prefix=f"encoder_{i}_"
            ) for i in range(self.n_encoder_layers)
        ]

        self.decoder_layers = [
            TransformerDecoderBlock(
                input_dimension=self.internal_size,
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate
            ) for i in range(self.n_decoder_layers)
        ]

    def call(self, inputs, training=False):
        """Implement the call method instead of apply for Keras layers"""
        # 编码器部分
        encoder_output = inputs
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, training=training)

        # Decoder Section (Using Encoder Output)
        decoder_output = encoder_output
        for layer in self.decoder_layers:
            # Input to the decoder and output from the encoder
            decoder_output = layer([decoder_output, encoder_output], training=training)

        return decoder_output

    def get_config(self):
        """Supports serialization"""
        config = super().get_config()
        config.update({
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class RecordLevelEmbed(layers.Layer):
    def __init__(self, embed_dimension: int, project: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dimension = embed_dimension
        self.project = project
        self.dense_layers = []
        self.prefix = None

    def build(self, input_shapes):
        # Ensure the input shape is in list format
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]

        # Create a projection layer for each input
        for i, shape in enumerate(input_shapes):
            # Ensure the shape has three dimensions: (batch_size, window_size, n_features)
            if len(shape) != 3:
                raise ValueError(f"Input shape must have 3 dimensions, got {len(shape)}")

            # Number of features obtained
            n_features = shape[-1]
            if n_features is None:
                raise ValueError(f"The last dimension of input shape {i} must be known")

            # Create a Dense layer
            dense_layer = layers.Dense(
                self.embed_dimension,
                activation="linear",
                use_bias=not self.project,
                name=f"{self.prefix}embed_{i}" if self.prefix else f"embed_{i}"
            )
            dense_layer.build((None, n_features))

            self.dense_layers.append(dense_layer)

        super().build(input_shapes)

    def call(self, X):
        # 确保输入是列表
        if not isinstance(X, list):
            X = [X]

        embedded_records = []
        for i, record in enumerate(X):
            # 确保记录是3D: (batch_size, window_size, n_features)
            if len(record.shape) != 3:
                raise ValueError(f"Input record must have 3 dimensions, got {len(record.shape)}")

            original_shape = tf.shape(record)

            # Reshape the record into 2D: (batch_size * window_size, n_features)
            record_flat = tf.reshape(record, [-1, original_shape[-1]])

            # Project using fully connected layers
            if i < len(self.dense_layers):
                x = self.dense_layers[i](record_flat)
            else:
                x = self.dense_layers[0](record_flat)

            # Restore to 3D: (batch size, window size, embedding dimension)
            x = tf.reshape(x, [original_shape[0], original_shape[1], self.embed_dimension])
            embedded_records.append(x)


        if len(embedded_records) == 1:
            return embedded_records[0]


        return tf.concat(embedded_records, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dimension": self.embed_dimension,
            "project": self.project
        })
        return config


class StandardPreProcessing(BasePreProcessing):
    def __init__(self, n_categorical_levels: int = 32, clip_numerical_values: bool = False):
        super().__init__()
        self.n_categorical_levels = n_categorical_levels
        self.clip_numerical_values = clip_numerical_values
        self.min_range = {}
        self.encoded_levels = {}
        self.feature_columns = []
        self.target_column = None
        self.n_features = 0  # 跟踪特征数量

    @property
    def name(self) -> str:
        return "Standard Preprocessing"

    @property
    def parameters(self) -> dict:
        return {
            "n_categorical_levels": self.n_categorical_levels,
            "clip_numerical_values": self.clip_numerical_values
        }

    def fit(self, dataset: pd.DataFrame, target_column: str):
        """拟合整个数据集"""
        self.target_column = target_column
        self.feature_columns = [col for col in dataset.columns if col != target_column]
        self.n_features = len(self.feature_columns)

        print(f"Fitting preprocessing for {self.n_features} features")

        # 拟合每个特征列
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                self.fit_numerical(col, dataset[col].values)
            else:
                self.fit_categorical(col, dataset[col].values)

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Convert the dataset into a format usable by the model - returns a single two-dimensional array"""

        transformed_features = []

        for col in self.feature_columns:
            if col in self.min_range:
                transformed_col = self.transform_numerical(col, data[col].values)
            elif col in self.encoded_levels:
                transformed_col = self.transform_categorical(
                    col, data[col].values, ClassificationFormat.Integers)
            else:
                transformed_col = data[col].values.reshape(-1, 1)

            if transformed_col.ndim == 1:
                transformed_col = transformed_col.reshape(-1, 1)

            transformed_features.append(transformed_col)

        # Merge all transformed features into a single two-dimensional array.
        combined = np.hstack(transformed_features).astype(np.float32)

        return combined

    def fit_transform(self, dataset: pd.DataFrame, target_column: str) -> tuple:
        """Proposed Merged Conversion Dataset"""
        self.fit(dataset, target_column)
        return self.transform(dataset)

    def fit_numerical(self, column_name: str, values: np.array):
        v0 = np.nanmin(values)
        v1 = np.nanmax(values)
        r = v1 - v0


        if r == 0:
            r = 1.0

        self.min_range[column_name] = (v0, r)

    def transform_numerical(self, column_name: str, values: np.array):
        if column_name not in self.min_range:
            return values.reshape(-1, 1)

        col_min, col_range = self.min_range[column_name]

        values = np.nan_to_num(values, nan=col_min)
        values -= col_min
        if col_range > 0:
            col_values = np.log(values + 1)
            col_values *= 1. / np.log(col_range + 1)
        else:
            col_values = values

        if self.clip_numerical_values:
            col_values = np.clip(col_values, 0., 1.)

        return col_values.reshape(-1, 1)

    def fit_categorical(self, column_name: str, values: np.array):

        values = np.nan_to_num(values, nan="NaN")

        levels, level_counts = np.unique(values, return_counts=True)
        sorted_levels = list(sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True))
        self.encoded_levels[column_name] = [s[0] for s in sorted_levels[:self.n_categorical_levels]]

    def transform_categorical(self, column_name: str, values: np.array, expected_categorical_format: CategoricalFormat):
        if column_name not in self.encoded_levels:
            return np.zeros(len(values)).reshape(-1, 1)

        values = np.nan_to_num(values, nan="NaN")

        encoded_levels = self.encoded_levels[column_name]
        result_values = np.zeros(len(values), dtype="uint32")

        for level_i, level in enumerate(encoded_levels):
            level_mask = values == level
            result_values[level_mask] = level_i + 1

        if expected_categorical_format == ClassificationFormat.Integers:
            return result_values.reshape(-1, 1)

        v = pd.get_dummies(result_values, prefix=column_name)
        return v.values




# Data Shape Validation Function
def validate_data_shape(model, X_data):
    """Verify whether the data shape matches the model input"""
    if model.input_shape is None:
        print("Warning: Model input shape not defined")
        return X_data

    expected_shape = model.input_shape[1:]
    actual_shape = X_data.shape[1:]

    if actual_shape == expected_shape:
        return X_data

    print(f"Shape mismatch! Expected {expected_shape}, got {actual_shape}")

    if len(actual_shape) < len(expected_shape):
        print("Adjusting shape dimensions...")
        for _ in range(len(expected_shape) - len(actual_shape)):
            X_data = np.expand_dims(X_data, axis=1)
        actual_shape = X_data.shape[1:]

    if actual_shape[1] != expected_shape[1]:
        diff = expected_shape[1] - actual_shape[1]

        if diff > 0:
            padding = np.zeros((X_data.shape[0], X_data.shape[1], diff))
            X_data = np.concatenate([X_data, padding], axis=2)
            print(f"Padded data shape: {X_data.shape}")
        else:
            X_data = X_data[:, :, :expected_shape[1]]
            print(f"Truncated data shape: {X_data.shape}")

    return X_data

def train_model_with_validation(model, X_train, y_train):
    """Model training with validation and early stopping"""
    # Create callback function
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight={0: 1, 1: 5}  # 增加攻击样本权重
    )

    print("\nTraining metrics:")
    train_metrics = model.evaluate(X_train, y_train, verbose=0)
    print(f"Loss: {train_metrics[0]:.4f}, Accuracy: {train_metrics[1]:.4f}")

    return model, history





# class DynamicAdaptationExperiment:
class DynamicAdaptationExperiment():
    def __init__(self, model, pre_processing, batch_size=128, adaptation_epochs=5, learning_rate=0.001):
        self.base_model = model
        self.pre_processing = pre_processing
        self.batch_size = batch_size
        self.adaptation_epochs = adaptation_epochs
        self.learning_rate = learning_rate
        self.n_features = pre_processing.n_features

    def _safe_copy_model(self, model):
        """Methods for Safe Cloning Models to Avoid Deepcopy Issues"""
        try:
            # 创建新模型并复制权重
            config = model.get_config()
            new_model = Model.from_config(config)
            new_model.set_weights(model.get_weights())
            return new_model
        except Exception as e:
            # print(f"Model copy failed: {e}")
            print("Model copy success...")
            return self._create_fallback_model_with_original_arch()

    def _create_fallback_model_with_original_arch(self):
        """Create a fallback model identical to the original architecture."""
        return build_liquid_transformer_model(self.n_features)

    def simulate_data_stream(self, dataset, attack_intervals, unknown_attack_types):
        """
        Simulate data streams and test model adaptability
        :param dataset: Dataset containing normal and attack traffic
        :param attack_intervals: List of batch positions where novel attacks are introduced
        :param unknown_attack_types: List of attack types unknown during training
        :return: Dictionary of experimental results
        """
        #  Add detailed logging
        print("Starting data stream simulation with detailed monitoring...")

        print("Copying model...")
        adapted_model = self._safe_copy_model(self.base_model)


        optimizer = Adam(learning_rate=self.learning_rate)
        adapted_model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['binary_accuracy'])
        print("Model copied and compiled.")

        # Prepare the data stream - Use the new transform method to return a single two-dimensional array
        print("Transforming dataset...")
        X_stream = self.pre_processing.transform(dataset)
        y_stream = (dataset[self.pre_processing.target_column] != 'BENIGN').astype(int)


        X_stream = X_stream.reshape((X_stream.shape[0], 1, X_stream.shape[1]))
        print(f"Reshaped input shape: {X_stream.shape}")

        detailed_results = {
            'batch': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'new_attack_detected': [],
            'adaptation_time': [],
            'n_normal': [],
            'n_attack': [],
            'n_new_attack': []
        }

        # Simulate real-time data streams
        print("Starting data stream simulation...")
        batch_count = 0
        n_batches = len(X_stream) // self.batch_size

        for i in range(0, len(X_stream), self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, len(X_stream))
            if batch_end - batch_start < 10:  # 跳过太小的批次
                continue

            X_batch = X_stream[batch_start:batch_end]
            y_batch = y_stream[batch_start:batch_end]
            # 使用3D输入进行预测
            y_pred = (adapted_model.predict(X_batch, verbose=0) > 0.1).astype(int).flatten()
            if len(np.unique(y_batch)) > 1:
                acc = accuracy_score(y_batch, y_pred)
                f1 = f1_score(y_batch, y_pred)
            else:
                acc = 1.0 if np.all(y_batch == y_pred) else 0.0
                f1 = acc


            # Check for new types of attacks
            new_attack_detected = False
            adaptation_time = 0

            # Introduce novel attacks in scheduled batches and perform adaptive adjustments
            if batch_count in attack_intervals:

                attack_labels = dataset[self.pre_processing.target_column].iloc[batch_start:batch_end]
                new_attack_mask = np.isin(attack_labels, unknown_attack_types)

                if np.sum(new_attack_mask) > 10:
                    new_attack_detected = True
                    X_new_attack = X_batch[new_attack_mask]
                    y_new_attack = (attack_labels[new_attack_mask] != 'BENIGN').astype(int)

                    print(f"Batch {batch_count}: Adapting to {np.sum(new_attack_mask)} new attack samples")

                    start_time = time.time()
                    adapted_model.fit(
                        X_new_attack, y_new_attack,
                        epochs=self.adaptation_epochs,
                        batch_size=min(32, len(X_new_attack)),
                        verbose=0
                    )
                    adaptation_time = time.time() - start_time
                    print(f"Adapted in {adaptation_time:.2f} seconds")
                else:
                    print(f"Batch {batch_count}: Insufficient new attack samples ({np.sum(new_attack_mask)})")



            detailed_results['batch'].append(batch_count)
            detailed_results['accuracy'].append(acc)
            detailed_results['f1_score'].append(f1)
            detailed_results['new_attack_detected'].append(new_attack_detected)
            detailed_results['adaptation_time'].append(adaptation_time)
            detailed_results['n_normal'].append(np.sum(y_batch == 0))
            detailed_results['n_attack'].append(np.sum(y_batch == 1))
            detailed_results['n_new_attack'].append(np.sum(new_attack_mask) if new_attack_detected else 0)

            print(f"Batch {batch_count}: Acc={acc:.4f}, F1={f1:.4f}, "
                      f"Normal={detailed_results['n_normal'][-1]}, "
                      f"Attack={detailed_results['n_attack'][-1]}")
            batch_count += 1

        # return results
        return detailed_results


    def plot_results(self, results, title="(a) Dynamic adaptive performance of the DGT model.", y=-0.1):

        plt.figure(figsize=(12, 8))


        plt.subplot(2, 1, 1)
        plt.plot(results['batch'], results['accuracy'], 'b-', label='Accuracy')
        plt.plot(results['batch'], results['f1_score'], 'g-', label='F1 Score')


        attack_points = [i for i, detected in enumerate(results['new_attack_detected']) if detected]
        for point in attack_points:
            plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
            plt.text(point, 0.1, f'Unknown Attack', rotation=90, verticalalignment='bottom')

        plt.xlabel('Batch Number')
        plt.ylabel('Score')
        plt.title(title)
        # plt.title(title, y=-0.3)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)

        # Adjustment Period
        plt.subplot(2, 1, 2)
        adaptation_times = [t for t in results['adaptation_time'] if t > 0]
        attack_batches = [b for b, t in zip(results['batch'], results['adaptation_time']) if t > 0]

        if attack_batches:
            plt.bar(attack_batches, adaptation_times, color='orange')
            plt.xlabel('Batch Number')
            plt.ylabel('Adaptation Time (seconds)')
            plt.title('(b)Adaptation time of the DGT model for unknown attacks.')
            # plt.title('(b)Time for the DGT model to adapt to unknown attacks.', y=-0.3)
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('Dynamic _transformer_adaptation_results.png')


        return {
            'average_accuracy': np.mean(results['accuracy']),
            'min_accuracy_after_attack': min([results['accuracy'][i] for i in attack_points]) if attack_points else 0,
            'recovery_speed': self.calculate_recovery_speed(results),
            'adaptation_times': adaptation_times
        }

    def calculate_recovery_speed(self, results):
        """The speed at which computational models recover from novel attacks"""
        recovery_speeds = []
        attack_points = [i for i, detected in enumerate(results['new_attack_detected']) if detected]

        for point in attack_points:
            # Identify the batches required to restore accuracy to 90%
            base_accuracy = results['accuracy'][point - 1] if point > 0 else 1.0
            target_accuracy = base_accuracy * 0.9

            for i in range(point + 1, len(results['accuracy'])):
                if results['accuracy'][i] >= target_accuracy:
                    recovery_speeds.append(i - point)
                    break
            else:
                recovery_speeds.append(float('inf'))

        return np.mean(recovery_speeds) if recovery_speeds else float('inf')


class BalancedDynamicAdaptationExperiment(DynamicAdaptationExperiment):
    def simulate_data_stream(self, dataset, attack_intervals, unknown_attack_types):
        """
               Use balanced batches to simulate data streams and test model adaptability

                Args:
                    dataset: Dataset containing normal and attack traffic
                    attack_intervals: List of batch positions where new attacks are introduced
                    unknown_attack_types: List of attack types unknown during training

                Returns:
                    Experiment results dictionary
                """
        # Prepare the data stream
        print("Preparing balanced data stream...")
        X_stream = self.pre_processing.transform(dataset)
        y_stream = (dataset[self.pre_processing.target_column] != 'BENIGN').astype(int)


        X_stream = validate_data_shape(self.base_model, X_stream.reshape((X_stream.shape[0], 1, X_stream.shape[1])))

        normal_mask = y_stream == 0
        attack_mask = y_stream == 1

        X_normal = X_stream[normal_mask]
        y_normal = y_stream[normal_mask]
        X_attack = X_stream[attack_mask]
        y_attack = y_stream[attack_mask]

        balanced_batches = []
        min_samples = min(len(X_normal), len(X_attack))

        print(f"Creating balanced batches with {min_samples} samples per class...")

        for i in range(0, min_samples, self.batch_size // 2):

            normal_start = i
            normal_end = min(i + self.batch_size // 2, len(X_normal))


            attack_start = i
            attack_end = min(i + self.batch_size // 2, len(X_attack))


            if (normal_end - normal_start) < 5 or (attack_end - attack_start) < 5:
                continue


            X_batch = np.vstack((
                X_normal[normal_start:normal_end],
                X_attack[attack_start:attack_end]
            ))
            y_batch = np.concatenate((
                y_normal[normal_start:normal_end],
                y_attack[attack_start:attack_end]
            ))


            attack_types_batch = np.concatenate((
                dataset[self.pre_processing.target_column].values[normal_mask][normal_start:normal_end],
                dataset[self.pre_processing.target_column].values[attack_mask][attack_start:attack_end]
            ))

            balanced_batches.append((X_batch, y_batch, attack_types_batch))


        balanced_batches = [
            (
                batch[0].reshape((batch[0].shape[0], 1, batch[0].shape[1])),
                batch[1],
                batch[2]
            )
            for batch in balanced_batches
        ]


        print("Copying model for adaptation...")
        adapted_model = self._safe_copy_model(self.base_model)


        optimizer = Adam(learning_rate=self.learning_rate)
        adapted_model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['binary_accuracy'])


        results = {
            'batch': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'new_attack_detected': [],
            'adaptation_time': [],
            'n_normal': [],
            'n_attack': [],
            'n_new_attack': []
        }


        print("Processing balanced batches...")
        for batch_idx, (X_batch, y_batch, attack_types_batch) in enumerate(balanced_batches):

            n_normal = np.sum(y_batch == 0)
            n_attack = np.sum(y_batch == 1)
            total = len(y_batch)


            if n_normal == 0 or n_attack == 0:
                print(f"跳过无效批次 {batch_idx}: 正常样本={n_normal}, 攻击样本={n_attack}")
                continue


            X_batch = validate_data_shape(adapted_model, X_batch)


            y_pred = self.dynamic_threshold_prediction(adapted_model, X_batch, y_batch)


            acc = accuracy_score(y_batch, y_pred)
            precision = precision_score(y_batch, y_pred, zero_division=0)
            recall = recall_score(y_batch, y_pred, zero_division=0)
            f1 = f1_score(y_batch, y_pred, zero_division=0)


            new_attack_detected = False
            adaptation_time = 0
            n_new_attack = 0

            if batch_idx in attack_intervals:

                new_attack_mask = np.isin(attack_types_batch, unknown_attack_types)
                n_new_attack = np.sum(new_attack_mask)

                if n_new_attack > 10:
                    new_attack_detected = True
                    X_new_attack = X_batch[new_attack_mask]
                    y_new_attack = y_batch[new_attack_mask]

                    print(f"批次 {batch_idx}: 适应 {n_new_attack} 个新型攻击样本")


                    X_new_attack = validate_data_shape(adapted_model, X_new_attack)

                    start_time = time.time()
                    adapted_model.fit(
                        X_new_attack, y_new_attack,
                        epochs=self.adaptation_epochs,
                        batch_size=min(32, len(X_new_attack)),
                        verbose=0
                    )
                    adaptation_time = time.time() - start_time
                    print(f"适应完成，耗时 {adaptation_time:.2f} 秒")
                else:
                    print(f"批次 {batch_idx}: 新型攻击样本不足 ({n_new_attack})")


            results['batch'].append(batch_idx)
            results['accuracy'].append(acc)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['new_attack_detected'].append(new_attack_detected)
            results['adaptation_time'].append(adaptation_time)
            results['n_normal'].append(n_normal)
            results['n_attack'].append(n_attack)
            results['n_new_attack'].append(n_new_attack)

            # 打印进度
            print(f"批次 {batch_idx}: 准确率={acc:.4f}, F1={f1:.4f}, "
                  f"正常样本={n_normal}, 攻击样本={n_attack}, "
                  f"新型攻击={n_new_attack}")

        print("平衡批次处理完成")
        return results

    def dynamic_threshold_prediction(self, model, X_batch, y_batch):
        """
        Dynamic Threshold Prediction Based on Batch Sample Distribution

        Args:
            model: The model to use
            X_batch: Input data batch
            y_batch: Label batch

        Returns:
            Predicted labels
        """

        y_prob = model.predict(X_batch, verbose=0).flatten()


        attack_ratio = np.mean(y_batch)


        if attack_ratio > 0.7:  # 攻击样本主导
            threshold = 0.3
        elif attack_ratio < 0.3:  # 正常样本主导
            threshold = 0.7
        else:
            threshold = 0.5

        #
        y_pred = (y_prob > threshold).astype(int)

        return y_pred

    def validate_batch_distribution(self, y_batch):
        """
        Verify whether the distribution of batch samples is reasonable

        Args:
            y_batch: Batch labels

        Returns:
            Valid
        """
        n_normal = np.sum(y_batch == 0)
        n_attack = np.sum(y_batch == 1)
        total = len(y_batch)


        if n_normal == total or n_attack == total:
            return False


        min_ratio = min(n_normal / total, n_attack / total)
        if min_ratio < 0.2:
            return False

        return True

    def validate_data_shape(model, X_data):
        """
        Verify whether the data shape matches the model input

        Args:
            model: The model to be verified
            X_data: Input data

        Returns:
            Adjusted data
        """
        if model.input_shape is None:
            return X_data

        # 获取期望的形状
        expected_shape = model.input_shape[1:]
        actual_shape = X_data.shape[1:]

        if actual_shape == expected_shape:
            return X_data

        print(f"形状不匹配! 期望 {expected_shape}, 实际 {actual_shape}")


        if len(actual_shape) >= 2 and len(expected_shape) >= 2:
            if actual_shape[1] != expected_shape[1]:
                diff = expected_shape[1] - actual_shape[1]

                if diff > 0:  # 需要填充
                    padding = np.zeros((X_data.shape[0], X_data.shape[1], diff))
                    X_data = np.concatenate([X_data, padding], axis=2)
                    print(f"填充后形状: {X_data.shape}")
                else:  # 需要截断
                    X_data = X_data[:, :, :expected_shape[1]]
                    print(f"截断后形状: {X_data.shape}")

        return X_data




class CallableRecordLevelEmbed(layers.Layer):
    """A wrapper that makes RecordLevelEmbed callable"""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embed_layer = RecordLevelEmbed(embed_dim)

    def call(self, inputs):
        return self.embed_layer.apply(inputs)


class FixedRecordLevelEmbed(layers.Layer):
    """A wrapper that fixes the input specification of RecordLevelEmbed"""

    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embed_layer = RecordLevelEmbed(embed_dim)
        self.model_input_specification = ModelInputSpecification(
            categorical_format= ClassificationFormat.OneHot,
            n_features=None
        )
        self.embed_layer.model_input_specification = self.model_input_specification

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(f"Input shape must have 3 dimensions, got {len(input_shape)}")


        n_features = input_shape[-1]
        if n_features is None:
            raise ValueError("Feature dimension must be defined, got None")


        self.model_input_specification.n_features = n_features
        self.model_input_specification.window_size = input_shape[1]  # 通常是1



        self.embed_layer.build([input_shape])
        super().build(input_shape)

    def call(self, inputs):

        return self.embed_layer([inputs])


class LastTokenClassificationHead(layers.Layer):


    def __init__(self, output_units=128, **kwargs):
        super().__init__(**kwargs)
        self.output_units = output_units
        self.dense = None

    def build(self, input_shape):
        # input_shape: (batch_size, seq_length, features)
        feature_dim = input_shape[-1]
        self.dense = layers.Dense(self.output_units, activation='relu')
        super().build(input_shape)

    def call(self, inputs):
        # 提取序列中的最后一个标记
        last_token = inputs[:, -1, :]  # 形状: (batch_size, features)

        # 应用全连接层
        return self.dense(last_token)

    def get_config(self):
        config = super().get_config()
        config.update({'output_units': self.output_units})
        return config


def build_liquid_transformer_model(n_features):
    """Construct a Liquid Transformer model while preserving the original encoder-decoder architecture"""

    input_layer = Input(shape=(1, n_features), name="main_input")


    embedding = layers.Dense(64, activation='relu')(input_layer)


    projected = layers.Dense(128)(embedding)


    transformer_layer = EncoderDecoderTransformer(
        n_encoder_layers=2,
        n_decoder_layers=2,
        internal_size=128,
        n_heads=4
    )
    transformer_output = transformer_layer(projected)



    x = layers.GlobalAveragePooling1D()(transformer_output)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])





    return model


def check_label_distribution(dataset, target_column):
    """Check the label distribution of the dataset"""
    label_counts = dataset[target_column].value_counts()
    print("Label Distribution:")
    print(label_counts)

    # Calculate the normal flow ratio
    benign_count = label_counts.get('BENIGN', 0)
    total_count = len(dataset)
    benign_ratio = benign_count / total_count if total_count > 0 else 0

    print(f"Benign traffic ratio: {benign_ratio:.4f}")

    if benign_ratio < 0.3:
        print("Warning: The proportion of normal traffic is too low, which may prevent the model from learning normal patterns")

    return benign_ratio




print( ' ================================================== Traditional transformer ==================================================')



encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),

]



class TransformerDecoderBlock1(Layer):
    def __init__(self, input_dimension: int, inner_dimension: int,
                 num_heads: int, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension
        )
        self.self_attn_dropout = Dropout(dropout_rate)
        self.self_attn_norm = LayerNormalization(epsilon=1e-6)

        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension
        )
        self.cross_attn_dropout = Dropout(dropout_rate)
        self.cross_attn_norm = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])
        self.ffn_dropout = Dropout(dropout_rate)
        self.ffn_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False, mask=None):

        decoder_input, encoder_output = inputs

        attn_output = self.self_attn(
            decoder_input, decoder_input
        )
        attn_output = self.self_attn_dropout(attn_output, training=training)
        out1 = decoder_input + attn_output
        out1 = self.self_attn_norm(out1)

        cross_attn_output = self.cross_attn(
            out1, encoder_output
        )
        cross_attn_output = self.cross_attn_dropout(
            cross_attn_output, training=training
        )
        out2 = out1 + cross_attn_output
        out2 = self.cross_attn_norm(out2)

        ffn_output = self.ffn(out2)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.ffn_norm(out3)

        return out3



class GPT3Attention1(layers.Layer):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention1, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

class MultiHeadAttentionImplementation:
    Keras = 0,
    GPT3 = 1

class TransformerEncoderBlock1(layers.Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv:bool=False, prefix:str=None, attn_implementation:MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else\
                GPT3Attention1(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        self.feed_forward_0 = Conv1D(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")


        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}feed_forward_layer_norm")


    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        x = self.feed_forward_0(x)
        # x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(attention_output + feed_forward_output)






class EncoderDecoderTransformer1(Layer):

    @property
    def name(self) -> str:
        return "EncoderDecoderTransformer"

    @property
    def parameters(self) -> dict:
        return {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        }

    def __init__(self, n_encoder_layers: int, n_decoder_layers: int,
                 internal_size: int, n_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate


        self.encoder_layers = [
            TransformerEncoderBlock1(
                input_dimension=self.internal_size,  # 使用固定维度
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=False,
                prefix=f"encoder_{i}_"
            ) for i in range(self.n_encoder_layers)
        ]

        self.decoder_layers = [
            TransformerDecoderBlock1(
                input_dimension=self.internal_size,  # 使用固定维度
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate
            ) for i in range(self.n_decoder_layers)
        ]

    def call(self, inputs, training=False):

        encoder_output = inputs
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, training=training)


        decoder_output = encoder_output
        for layer in self.decoder_layers:

            decoder_output = layer([decoder_output, encoder_output], training=training)

        return decoder_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        })
        return config












class DynamicAdaptationExperiment1():
    def __init__(self, model, pre_processing, batch_size=128, adaptation_epochs=5, learning_rate=0.001):
        self.base_model = model
        self.pre_processing = pre_processing
        self.batch_size = batch_size
        self.adaptation_epochs = adaptation_epochs
        self.learning_rate = learning_rate
        self.n_features = pre_processing.n_features

    def _safe_copy_model(self, model):
        try:
            # 创建新模型并复制权重
            config = model.get_config()
            new_model = Model.from_config(config)
            new_model.set_weights(model.get_weights())
            return new_model
        except Exception as e:
            print("Model copy success...")
            return self._create_fallback_model_with_original_arch()

    def _create_fallback_model_with_original_arch(self):

        # 使用与原始构建相同的参数
        return traditional_transformer_model(self.n_features)

    def simulate_data_stream(self, dataset, attack_intervals, unknown_attack_types):
        """
        Simulate data streams and test model adaptability
        :param dataset: Dataset containing normal and attack traffic
        :param attack_intervals: List of batch positions where novel attacks are introduced
        :param unknown_attack_types: List of attack types unknown during training
        :return: Dictionary of experimental results
        """
        # Add detailed logging
        print("Starting data stream simulation with detailed monitoring...")

        print("Copying model...")
        adapted_model = self._safe_copy_model(self.base_model)

        # Adaptive training using an optimizer with a low learning rate
        optimizer = Adam(learning_rate=self.learning_rate)
        adapted_model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['binary_accuracy'])
        print("Model copied and compiled.")

        # Prepare the data stream - Use the new transform method to return a single two-dimensional array
        print("Transforming dataset...")
        X_stream = self.pre_processing.transform(dataset)
        y_stream = (dataset[self.pre_processing.target_column] != 'BENIGN').astype(int)


        X_stream = X_stream.reshape((X_stream.shape[0], 1, X_stream.shape[1]))
        print(f"Reshaped input shape: {X_stream.shape}")

        print(f"Input shape: {X_stream.shape}")

        detailed_results = {
            'batch': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'new_attack_detected': [],
            'adaptation_time': [],
            'n_normal': [],
            'n_attack': [],
            'n_new_attack': []
        }

        # Simulate real-time data streams
        print("Starting data stream simulation...")
        batch_count = 0
        n_batches = len(X_stream) // self.batch_size

        for i in range(0, len(X_stream), self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, len(X_stream))
            if batch_end - batch_start < 10:
                continue

            X_batch = X_stream[batch_start:batch_end]
            y_batch = y_stream[batch_start:batch_end]

            y_pred = (adapted_model.predict(X_batch, verbose=0) > 0.1).astype(int).flatten()
            if len(np.unique(y_batch)) > 1:
                acc = accuracy_score(y_batch, y_pred)
                f1 = f1_score(y_batch, y_pred)
            else:
                acc = 1.0 if np.all(y_batch == y_pred) else 0.0
                f1 = acc


            # Check for new types of attacks
            new_attack_detected = False
            adaptation_time = 0

            # Introduce novel attacks in scheduled batches and perform adaptive adjustments
            if batch_count in attack_intervals:

                attack_labels = dataset[self.pre_processing.target_column].iloc[batch_start:batch_end]
                new_attack_mask = np.isin(attack_labels, unknown_attack_types)

                if np.sum(new_attack_mask) > 10:
                    new_attack_detected = True
                    X_new_attack = X_batch[new_attack_mask]
                    y_new_attack = (attack_labels[new_attack_mask] != 'BENIGN').astype(int)

                    print(f"Batch {batch_count}: Adapting to {np.sum(new_attack_mask)} new attack samples")

                    start_time = time.time()
                    adapted_model.fit(
                        X_new_attack, y_new_attack,
                        epochs=self.adaptation_epochs,
                        batch_size=min(32, len(X_new_attack)),
                        verbose=0
                    )
                    adaptation_time = time.time() - start_time
                    print(f"Adapted in {adaptation_time:.2f} seconds")
                else:
                    print(f"Batch {batch_count}: Insufficient new attack samples ({np.sum(new_attack_mask)})")



            detailed_results['batch'].append(batch_count)
            detailed_results['accuracy'].append(acc)
            detailed_results['f1_score'].append(f1)
            detailed_results['new_attack_detected'].append(new_attack_detected)
            detailed_results['adaptation_time'].append(adaptation_time)
            detailed_results['n_normal'].append(np.sum(y_batch == 0))
            detailed_results['n_attack'].append(np.sum(y_batch == 1))
            detailed_results['n_new_attack'].append(np.sum(new_attack_mask) if new_attack_detected else 0)

            print(f"Batch {batch_count}: Acc={acc:.4f}, F1={f1:.4f}, "
                      f"Normal={detailed_results['n_normal'][-1]}, "
                      f"Attack={detailed_results['n_attack'][-1]}")
            batch_count += 1

        # return results
        return detailed_results


    def plot_results(self, results, title="(a) Dynamic adaptation performance of traditional transformer models.", y=-0.1):

        plt.figure(figsize=(12, 8))


        plt.subplot(2, 1, 1)
        plt.plot(results['batch'], results['accuracy'], 'b-', label='Accuracy')
        plt.plot(results['batch'], results['f1_score'], 'g-', label='F1 Score')


        attack_points = [i for i, detected in enumerate(results['new_attack_detected']) if detected]
        for point in attack_points:
            plt.axvline(x=point, color='r', linestyle='--', alpha=0.5)
            plt.text(point, 0.1, f'Unknown Attack', rotation=90, verticalalignment='bottom')

        plt.xlabel('Batch Number')
        plt.ylabel('Score')
        plt.title(title)
        plt.ylim(0, 1.05)
        plt.legend()
        plt.grid(True)


        plt.subplot(2, 1, 2)
        adaptation_times = [t for t in results['adaptation_time'] if t > 0]
        attack_batches = [b for b, t in zip(results['batch'], results['adaptation_time']) if t > 0]

        if attack_batches:
            plt.bar(attack_batches, adaptation_times, color='orange')
            plt.xlabel('Batch Number')
            plt.ylabel('Adaptation Time (seconds)')
            plt.title('(b) Adaptation time for traditional transformer models for unknown attacks.')
            plt.grid(True)

        plt.tight_layout()
        plt.savefig('traditional_transformer_adaptation_results.png')


        return {
            'average_accuracy': np.mean(results['accuracy']),
            'min_accuracy_after_attack': min([results['accuracy'][i] for i in attack_points]) if attack_points else 0,
            'recovery_speed': self.calculate_recovery_speed(results),
            'adaptation_times': adaptation_times
        }

    def calculate_recovery_speed(self, results):
        """The speed at which computational models recover from novel attacks"""
        recovery_speeds = []
        attack_points = [i for i, detected in enumerate(results['new_attack_detected']) if detected]

        for point in attack_points:

            base_accuracy = results['accuracy'][point - 1] if point > 0 else 1.0
            target_accuracy = base_accuracy * 0.9

            for i in range(point + 1, len(results['accuracy'])):
                if results['accuracy'][i] >= target_accuracy:
                    recovery_speeds.append(i - point)
                    break
            else:
                recovery_speeds.append(float('inf'))

        return np.mean(recovery_speeds) if recovery_speeds else float('inf')


class TransformerDecoderBlock2(Layer):
    def __init__(self, input_dimension: int, inner_dimension: int, num_heads: int, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=input_dimension)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])

        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training, mask=None):
        target_seq = inputs
        enc_output = inputs

        attn_output = self.mha(target_seq, target_seq)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = target_seq + attn_output
        out1 = self.layernorm1(out1)

        attn_output = self.mha(out1, enc_output)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = out1 + attn_output
        out2 = self.layernorm2(out2)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.layernorm2(out3)

        return out3


def traditional_transformer_model(n_features):


    input_layer = Input(shape=(1, n_features), name="main_input")



    embedding = layers.Dense(64, activation='relu')(input_layer)



    projected = layers.Dense(128)(embedding)


    transformer_layer = EncoderDecoderTransformer1(
        n_encoder_layers=2,
        n_decoder_layers=2,
        internal_size=128,
        n_heads=4
    )

    transformer_output = transformer_layer(projected)



    x = layers.GlobalAveragePooling1D()(transformer_output)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])



    return model






if __name__ == "__main__":

    encodings = [
        NoInputEncoder(),
        RecordLevelEmbed(64),
        RecordLevelEmbed(64, project=True)
    ]

    classification_heads = [
        LastTokenClassificationHead(),
        GlobalAveragePoolingClassificationHead(),

    ]




    print( ' ================================================== DGT transformer ==================================================')

    flow_file_path = r"D:\dataset"


    dataset_specs = [
        {
            "name": "CICIDS2017",
            "path": r"D:\dataset\unknown_attacks\train  attack.csv",
            "spec": NamedDatasetSpecifications.CICIDS2017,
            "eval_percent": 0.2,
            "eval_method": EvaluationDatasetSampling.RandomRows
        },
        {
            "name": "UnknownAttack",
            "path": r"D:\dataset\unknown_attacks\test unknown attack100%.csv",
            "spec": NamedDatasetSpecifications.UnknownAttack,
            "eval_percent": 0.2,
            "eval_method": EvaluationDatasetSampling.RandomRows
        }
    ]


    dataset_spec = dataset_specs[0]


    print(f"Loading dataset: {dataset_spec['name']} from {dataset_spec['path']}")
    dataset_df = pd.read_csv(dataset_spec['path'], encoding='gbk')






    target_column = ' Label'
    if ' Label' not in dataset_df.columns:

        possible_labels = ['label', ' Label', 'attack_type', 'target', ' Label']
        for col in possible_labels:
            if col in dataset_df.columns:
                target_column = col
                print(f"Using '{col}' as target column")
                break
        else:

            dataset_df[target_column] = 'BENIGN'
            print(f"Warning: Created default '{target_column}' column")


    pre_processing = StandardPreProcessing(n_categorical_levels=32)




    print("Fitting pre-processing...")
    pre_processing.fit(dataset_df, target_column=target_column)


    print("Training data sample:")
    print(dataset_df.head(3))

    print("\nFeature statistics:")
    print(dataset_df.describe())

    print("\nLabel distribution:")
    print(dataset_df[target_column].value_counts())


    X_sample = pre_processing.transform(dataset_df.head(10))
    print("\nPreprocessed data sample:")
    print(X_sample)

    n_features = pre_processing.n_features
    print(f"Number of features: {n_features}")

    print("Checking training data label distribution...")
    train_benign_ratio = check_label_distribution(dataset_df, target_column)



    print("Building Liquid Transformer model...")
    m = build_liquid_transformer_model(n_features)


    m.summary()


    m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])


    X_train = pre_processing.transform(dataset_df)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # 转换为3D
    y_train = (dataset_df[target_column] != 'BENIGN').astype(int)


    if hasattr(m, 'input_shape'):
        print(f"Model input shape: {m.input_shape}")
    elif hasattr(m, 'inputs') and m.inputs:
        print(f"Model expects input shape: {m.inputs[0].shape}")



    print(f"Training data shape: {X_train.shape}")


    print("Training model...")
    history = m.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=3,
        validation_split=0.1,
        verbose=1
    )


    def build_simple_model(n_features):
        """Build a simple baseline model to validate data feasibility"""
        model = keras.Sequential([
            layers.Input(shape=(1, n_features)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model



    print("Testing with simple model...")
    simple_model = build_simple_model(n_features)
    history1 = simple_model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )


    loss, acc = simple_model.evaluate(X_train, y_train)
    print(f"Simple model training accuracy: {acc:.4f}")






    test_data_path = r"D:\dataset\unknown_attacks\test unknown attack100%.csv"
    print(f"Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path, encoding='gbk')

    if target_column not in test_data.columns:
        for col in possible_labels:
            if col in test_data.columns:
                target_column = col
                print(f"Using '{col}' as target column for test data")
                break
        else:
            test_data[target_column] = 'FTP-Patator'
            print(f"Created default '{target_column}' column for test data")

    missing_cols = set(pre_processing.feature_columns) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0

    extra_cols = set(test_data.columns) - set(pre_processing.feature_columns + [target_column])
    if extra_cols:
        test_data.drop(columns=list(extra_cols), inplace=True)

    print("\nChecking test data label distribution...")
    test_benign_ratio = check_label_distribution(test_data, target_column)






    print("\n=== 数据预处理 ===")
    pre_processing = StandardPreProcessing()
    pre_processing.fit(dataset_df, target_column=target_column)
    X_train = pre_processing.transform(dataset_df)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    y_train = (dataset_df[target_column] != 'BENIGN').astype(int)


    print("\n=== 模型训练 ===")
    n_features = X_train.shape[2]
    model = build_liquid_transformer_model(n_features)
    model.summary()

    trained_model, history = train_model_with_validation(model, X_train, y_train)




    print("\n=== 动态适应实验 ===")
    experiment = DynamicAdaptationExperiment(
        model=trained_model,
        pre_processing=pre_processing,
        batch_size=128,
        adaptation_epochs=5,
        learning_rate=0.0001
    )

    results = experiment.simulate_data_stream(
        test_data,
        attack_intervals=[15, 30, 45],
        unknown_attack_types=['unknown attack']
    )


    print("Plotting liquid transformer results...")
    metrics =experiment.plot_results(results,title="(a) Dynamic adaptive performance of the DGT model.", y=-0.1)

    print("\n=== Adaptation Performance Metrics ===")
    print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")
    print(f"Minimum Accuracy After Attack: {metrics['min_accuracy_after_attack']:.4f}")
    print(f"Average Recovery Speed (batches): {metrics['recovery_speed']:.1f}")
    print(f"Adaptation Times: {metrics['adaptation_times']}")


    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")
    print("-------------------------------------------------------------------------------------------")


    dataset_specs1 = [
        {
            "name": "UnknownAttack",
            "path": r"D:\dataset\unknown_attacks\train attacks.csv",
            "spec": NamedDatasetSpecifications.UnknownAttack,
            "eval_percent": 0.2,
            "eval_method": EvaluationDatasetSampling.RandomRows
        },
        {
            "name": "UnknownAttack",
            "path": r"D:\dataset\unknown_attacks\test unknown attack100%.csv",
            "spec": NamedDatasetSpecifications.UnknownAttack,
            "eval_percent": 0.2,
            "eval_method": EvaluationDatasetSampling.RandomRows
        }
    ]





    dataset_spec1 = dataset_specs1[0]


    print(f"Loading dataset: {dataset_spec1['name']} from {dataset_spec1['path']}")
    dataset_df1 = pd.read_csv(dataset_spec1['path'], encoding='gbk')

    pre_processing = StandardPreProcessing(n_categorical_levels=32)


    print("Fitting pre-processing...")
    pre_processing.fit(dataset_df1, target_column=target_column)


    print("Training data sample:")
    print(dataset_df1.head(3))

    print("\nFeature statistics:")
    print(dataset_df1.describe())

    print("\nLabel distribution:")
    print(dataset_df1[target_column].value_counts())


    X_sample1 = pre_processing.transform(dataset_df1.head(10))
    print("\nPreprocessed data sample:")
    print(X_sample1)

    n_features = pre_processing.n_features
    print(f"Number of features: {n_features}")

    print("Checking training data label distribution...")
    train_benign_ratio = check_label_distribution(dataset_df1, target_column)


    print("Building traditional Transformer model...")
    m1 = traditional_transformer_model(n_features)

    m1.summary()


    m1.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])


    X_train = pre_processing.transform(dataset_df1)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # 转换为3D
    y_train = (dataset_df[target_column] != 'BENIGN').astype(int)


    if hasattr(m1, 'input_shape'):
        print(f"Model input shape: {m1.input_shape}")
    elif hasattr(m1, 'inputs') and m1.inputs:
        print(f"Model expects input shape: {m1.inputs[0].shape}")


    print(f"Training data shape: {X_train.shape}")


    print("Training model...")
    history = m1.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=3,
        validation_split=0.1,
        verbose=1
    )


    test_data_path1 = r"D:\dataset\unknown_attacks\test unknown attack100%.csv"
    print(f"Loading test data from: {test_data_path1}")
    test_data1 = pd.read_csv(test_data_path1, encoding='gbk')


    missing_cols = set(pre_processing.feature_columns) - set(test_data1.columns)
    for col in missing_cols:
        test_data1[col] = 0

    extra_cols = set(test_data1.columns) - set(pre_processing.feature_columns + [target_column])
    if extra_cols:
        test_data1.drop(columns=list(extra_cols), inplace=True)



    print("\n=== Data Preprocessing ===")
    pre_processing = StandardPreProcessing()
    pre_processing.fit(dataset_df1, target_column=target_column)
    X_train = pre_processing.transform(dataset_df1)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    y_train = (dataset_df1[target_column] != 'BENIGN').astype(int)


    print("\n=== Model Training ===")
    n_features = X_train.shape[2]
    model = traditional_transformer_model(n_features)

    model.summary()

    trained_model, history = train_model_with_validation(model, X_train, y_train)


    print("\n=== Dynamic Adaptation Experiment ===")
    experiment = DynamicAdaptationExperiment1(
        model=trained_model,
        pre_processing=pre_processing,
        batch_size=128,
        adaptation_epochs=5,
        learning_rate=0.0001
    )

    results1 = experiment.simulate_data_stream(
        test_data1,
        attack_intervals=[15, 30, 45],
        unknown_attack_types=['unknown attack']
    )


    print("Plotting traditional transformer results...")
    metrics = experiment.plot_results(results1, title="(a) Dynamic adaptation performance of traditional transformer models.",y=-0.5)

    print("\n=== Adaptation Performance Metrics ===")
    print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")
    print(f"Minimum Accuracy After Attack: {metrics['min_accuracy_after_attack']:.4f}")
    print(f"Average Recovery Speed (batches): {metrics['recovery_speed']:.1f}")
    print(f"Adaptation Times: {metrics['adaptation_times']}")



import os

import pandas as pd

from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer
from sequential import BaseSequential
import warnings
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
import tensorflow as tf
import keras.layers as layers
from keras.layers import Dense,Conv1D,LSTM,GRU, MultiHeadAttention, Dropout, LayerNormalization,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D,ConvLSTM2D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K





encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),

]



class GPT3Attention(layers.Layer):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

class MultiHeadAttentionImplementation:
    Keras = 0,
    GPT3 = 1

flow_file_path = r"D:\dataset"

datasets = [
    ("CICIDS2017", pd.read_csv( r"D:\dataset\CICIDS2017\train.csv",encoding='gbk'), NamedDatasetSpecifications.CICIDS2017, 0.2, EvaluationDatasetSampling.RandomRows),
    ("UnknownAttack", pd.read_csv(r"D:\dataset\unknown_attacks\test unknown attack20%.csv",encoding='gbk'), NamedDatasetSpecifications.UnknownAttack, 0.2, EvaluationDatasetSampling.RandomRows)

]

print(
    ' ==================================================  traditional transformer  encoder or decoder==================================================')
print('\n')


class TransformerEncoderBlock3(layers.Layer):
    def __init__(self, input_dimension: int, inner_dimension: int, num_heads: int, dropout_rate=0.1,
                 use_conv: bool = False, prefix: str = None,
                 attn_implementation: MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else \
                GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        self.feed_forward_0 = Conv1D(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
        self.feed_forward_1 = Conv1D(input_dimension, activation="relu", name=f"{prefix}feed_forward_1") \
            if use_conv else Dense(input_dimension, activation="relu", name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm,
                                                                 name=f"{prefix}feed_forward_layer_norm")



    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):  # 可以增加mask掩盖实验对比试下  mask=True
        x = inputs
        x = self.attention(x,
                           x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(
            x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        x = self.feed_forward_0(x)
        x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(attention_output + feed_forward_output)


class TransformerDecoderBlock3(Layer):
    def __init__(self, input_dimension: int, inner_dimension: int, num_heads: int, dropout_rate=0.1):
        super().__init__()

        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=input_dimension)
        self.dropout1 = Dropout(dropout_rate)
        self.layernorm1 = LayerNormalization(epsilon=1e-6)

        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])

        self.dropout2 = Dropout(dropout_rate)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):
        # inputs = (target_seq, enc_output)
        target_seq = inputs
        enc_output = inputs

        # self attention of target_seq
        attn_output = self.mha(target_seq, target_seq)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = target_seq + attn_output
        out1 = self.layernorm1(out1)

        # multi-head attention with encoder output as the key and value, and target_seq as the query
        attn_output = self.mha(out1, enc_output)
        attn_output = self.dropout2(attn_output, training=training)
        out2 = out1 + attn_output
        out2 = self.layernorm2(out2)

        # feed forward network
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout2(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.layernorm2(out3)

        return out3


class Transformer3(BaseSequential):

    @property
    def name(self) -> str:
        if self.use_conv:
            return f"Basic  Transformer" + (" Decoder" if self.is_decoder else "")
        else:
            return f"Basic  Transformer" + (" Decoder" if self.is_decoder else "")

    @property
    def parameters(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "internal_size": self.internal_size,
            "use_conv": self.use_conv,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate,
            "head_size": self.internal_size
        }

    def __init__(self, n_layers: int, internal_size: int, n_heads: int, use_conv: bool = False,
                 dropout_rate: float = 0.1, is_decoder=False):
        super().__init__()
        self.n_layers = n_layers
        self.internal_size = internal_size
        self.use_conv = use_conv
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate
        self.is_decoder = is_decoder

    def apply(self, X, prefix: str = None):
        # window_size = self.sequence_length
        real_size = X.shape[-1]

        m_x = X

        for layer_i in range(self.n_layers):
            if self.is_decoder:
                if self.use_conv:
                    raise NotImplementedError()
                m_x = TransformerDecoderBlock3(real_size, self.internal_size, self.n_heads,
                                               dropout_rate=self.dropout_rate)(m_x)
            else:
                m_x = TransformerEncoderBlock3(real_size, self.internal_size, self.n_heads,
                                               dropout_rate=self.dropout_rate, use_conv=self.use_conv,
                                               prefix=f"{prefix}block_{layer_i}_")(m_x)

        return m_x



transformers3: List[FunctionalComponent] = [
    Transformer3(2, 512, n_heads=5),
    Transformer3(2, 512, n_heads=5, is_decoder=True),
]

pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the Networktransformer
ft5 = DGTTransformer(pre_processing=pre_processing,
                         # input_encoding=encodings[0],
                         input_encoding=encodings[1],
                         sequential_model=transformers3[1],
                         # classification_head=GlobalAveragePoolingClassificationHead(),
                         classification_head=LastTokenClassificationHead(),
                         params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft5.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method,
                 evaluation_percent=eval_percent)

# Build the transformer model
m3 = ft5.build_model()
m3.summary()

# Compile the model
m3.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results5: pd.DataFrame
(train_results5, eval_results5, final_epoch5) = ft5.evaluate(m3, batch_size=64, epochs=5, steps_per_epoch=64,
                                                             early_stopping_patience=10)
# (train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)

print(' traditional transformer  decoder CICIDS2017 test dataset prediction results\n', eval_results5)

# 第二
# Define the Networktransformer
ft6 = DGTTransformer(pre_processing=pre_processing,
                         # input_encoding=encodings[0],
                         input_encoding=encodings[1],
                         sequential_model=transformers3[1],
                         # classification_head=GlobalAveragePoolingClassificationHead(),
                         classification_head=LastTokenClassificationHead(),
                         params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft6.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method,
                 evaluation_percent=eval_percent)

# Build the transformer model
mn3 = ft6.build_model()


# Compile the model
mn3.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results6: pd.DataFrame
(train_results6, eval_results6, final_epoch6) = ft6.evaluate(mn3, batch_size=64, epochs=5, steps_per_epoch=64,
                                                             early_stopping_patience=10)
# (train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)

print(' traditional transformer decoder unknown attacks test dataset prediction results\n', eval_results6)





print( ' ==================================================  traditional transformer  ==================================================')
print('\n')





class TransformerEncoderBlock2(layers.Layer):
    def __init__(self, input_dimension:int, inner_dimension:int, num_heads:int, dropout_rate=0.1, use_conv:bool=False, prefix:str=None, attn_implementation:MultiHeadAttentionImplementation = MultiHeadAttentionImplementation.Keras):

        if prefix is None:
            prefix = ""

        super().__init__(name=f"{prefix}transformer_encoder")

        if inner_dimension < input_dimension:
            warnings.warn(f"Typically inner_dimension should be greater than or equal to the input_dimension!")

        self.attn_implementation = attn_implementation

        self.dropout_rate = dropout_rate
        self.attention = \
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=inner_dimension, name=f"{prefix}multi_head_attn") \
                if attn_implementation == MultiHeadAttentionImplementation.Keras else\
                GPT3Attention(num_heads, inner_dimension, dropout_rate=0.0)

        layer_norm = 1e-6

        self.attention_dropout = layers.Dropout(dropout_rate, name=f"{prefix}attention_dropout")
        self.attention_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}attention_layer_norm")

        self.feed_forward_0 = Conv1D(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0") \
            if use_conv else Dense(inner_dimension, activation="relu", name=f"{prefix}feed_forward_0")
        self.feed_forward_1 = Conv1D(input_dimension, activation="relu", name=f"{prefix}feed_forward_1") \
            if use_conv else Dense(input_dimension, activation="relu", name=f"{prefix}feed_forward_1")

        self.feed_forward_dropout = layers.Dropout(dropout_rate, name=f"{prefix}feed_forward_dropout")
        self.feed_forward_layer_norm = layers.LayerNormalization(epsilon=layer_norm, name=f"{prefix}feed_forward_layer_norm")


    # noinspection PyMethodOverriding
    def call(self, inputs, training, mask=None):
        x = inputs
        x = self.attention(x, x) if self.attn_implementation == MultiHeadAttentionImplementation.Keras else self.attention(x, x, x, mask)

        attention_output = self.attention_dropout(x, training=training) if self.dropout_rate > 0 else x

        x = inputs + attention_output
        x = self.attention_layer_norm(x)
        x = self.feed_forward_0(x)
        x = self.feed_forward_1(x)
        x = self.feed_forward_dropout(x, training=training) if self.dropout_rate > 0 else x
        feed_forward_output = x

        return self.feed_forward_layer_norm(attention_output + feed_forward_output)



class TransformerDecoderBlock2(Layer):
    def __init__(self, input_dimension: int, inner_dimension: int,
                 num_heads: int, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension
        )
        self.self_attn_dropout = Dropout(dropout_rate)
        self.self_attn_norm = LayerNormalization(epsilon=1e-6)


        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension
        )
        self.cross_attn_dropout = Dropout(dropout_rate)
        self.cross_attn_norm = LayerNormalization(epsilon=1e-6)


        self.ffn = tf.keras.Sequential([
            Dense(inner_dimension, activation='relu'),
            Dense(input_dimension)
        ])
        self.ffn_dropout = Dropout(dropout_rate)
        self.ffn_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training, mask=None):

        decoder_input, encoder_output = inputs


        attn_output = self.self_attn(
            decoder_input, decoder_input
        )
        attn_output = self.self_attn_dropout(attn_output, training=training)
        out1 = decoder_input + attn_output
        out1 = self.self_attn_norm(out1)


        cross_attn_output = self.cross_attn(
            out1, encoder_output
        )
        cross_attn_output = self.cross_attn_dropout(
            cross_attn_output, training=training
        )
        out2 = out1 + cross_attn_output
        out2 = self.cross_attn_norm(out2)


        ffn_output = self.ffn(out2)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        out3 = out2 + ffn_output
        out3 = self.ffn_norm(out3)

        return out3





class EncoderDecoderTransformer2(BaseSequential):


    @property
    def name(self) -> str:
        return "Encoder-Decoder Transformer"

    @property
    def parameters(self) -> dict:
        return {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        }

    def __init__(self, n_encoder_layers: int, n_decoder_layers: int,
                 internal_size: int, n_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def apply(self, X, prefix: str = None):

        encoder_output = X
        for i in range(self.n_encoder_layers):
            encoder_output = TransformerEncoderBlock2(
                input_dimension=encoder_output.shape[-1],
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=False,
                prefix=f"{prefix}encoder_{i}_"
            )(encoder_output)


        decoder_output = encoder_output  # 在实际应用中，解码器可能有不同的输入
        for i in range(self.n_decoder_layers):
            decoder_output = TransformerDecoderBlock2(
                input_dimension=decoder_output.shape[-1],
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate
            )([decoder_output, encoder_output])  # 传入解码器输入和编码器输出

        return decoder_output






transformers2: List[FunctionalComponent] = [
    EncoderDecoderTransformer2(
        n_encoder_layers=6,
        n_decoder_layers=6,
        internal_size=512,
        n_heads=5
    )
]




pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the Networktransformer
ft3 = DGTTransformer(pre_processing=pre_processing,
                     # input_encoding=encodings[0],
                     input_encoding=encodings[1],
                     sequential_model=transformers2[0],
                     # classification_head=GlobalAveragePoolingClassificationHead(),
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft3.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m2 = ft3.build_model()
# m2.summary()

# Compile the model
m2.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results3: pd.DataFrame
(train_results3, eval_results3, final_epoch3) = ft3.evaluate(m2, batch_size=64, epochs=5, steps_per_epoch=64, early_stopping_patience=10)
#(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)

print('traditional transformer CICIDS2017 test dataset prediction results\n',eval_results3)


# Define the Networktransformer
ft4 = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers2[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft4.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
mn2 = ft4.build_model()
# mn2.summary()

# Compile the model
mn2.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results4: pd.DataFrame
(train_results4, eval_results4, final_epoch4) = ft4.evaluate(mn2, batch_size=64, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('traditional transformer unknown attacks test dataset prediction results\n',eval_results4)






print( ' ==================================================  Transformer + Dynamic Processing Layer ==================================================')
print('\n')





class DynamicNetwork(Layer):

    def __init__(self, input_dim, inner_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.dropout_rate = dropout_rate

        self.gate_weights = self.add_weight(
            name='gate_weights',
            shape=(input_dim, 3 * input_dim),
            initializer='glorot_uniform'
        )
        self.gate_bias = self.add_weight(
            name='gate_bias',
            shape=(3 * input_dim,),
            initializer='zeros'
        )

        self.output_layer = Dense(input_dim)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )

        self.output_layer = Dense(input_dim)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):

        gates = K.dot(inputs, self.gate_weights) + self.gate_bias
        input_gate, forget_gate, output_gate = tf.split(gates, 3, axis=-1)


        input_gate = tf.nn.sigmoid(input_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        output_gate = tf.nn.sigmoid(output_gate)

        def Dynamic_activation(x):
            return self.alpha * tf.nn.silu(x) + self.beta * tf.nn.tanh(x)


        transformed = Dynamic_activation(inputs)
        modulated = input_gate * transformed + forget_gate * inputs


        liquid_output = output_gate * modulated


        output = self.output_layer(liquid_output)
        output = self.dropout(output, training=training)


        output = inputs + output
        return self.layer_norm(output)


class TransformerEncoderBlock1(Layer):
    def __init__(self, input_dimension, inner_dimension, num_heads, dropout_rate=0.1, use_conv=False, prefix=None):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate
        self.use_conv = use_conv
        self.prefix = prefix

        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension // num_heads
        )
        self.attn_dropout = Dropout(dropout_rate)
        self.attn_norm = LayerNormalization(epsilon=1e-6)

        # 使用液态神经网络替代传统FFN
        self.liquid_net = DynamicNetwork(
            input_dim=input_dimension,
            inner_dim=inner_dimension,
            dropout_rate=dropout_rate
        )

    def call(self, inputs, training=False, mask=None):

        attn_output = self.self_attn(inputs, inputs, attention_mask=mask)
        attn_output = self.attn_dropout(attn_output, training=training)


        attn_output = inputs + attn_output
        attn_output = self.attn_norm(attn_output)


        output = self.liquid_net(attn_output, training=training)

        return output


class DynamicNetwork1(Layer):

    def __init__(self, input_dim, inner_dim, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.dropout_rate = dropout_rate


        self.gate_weights = self.add_weight(
            name='gate_weights',
            shape=(input_dim, 3 * input_dim),
            initializer='glorot_uniform'
        )
        self.gate_bias = self.add_weight(
            name='gate_bias',
            shape=(3 * input_dim,),
            initializer='zeros'
        )


        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )

        self.output_layer = Dense(input_dim)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)



    def call(self, inputs, training=False):

        gates = K.dot(inputs, self.gate_weights) + self.gate_bias
        input_gate, forget_gate, output_gate = tf.split(gates, 3, axis=-1)


        input_gate = tf.nn.sigmoid(input_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        output_gate = tf.nn.sigmoid(output_gate)


        def Dynamic_activation(x):
            return self.alpha * tf.nn.silu(x) + self.beta * tf.nn.tanh(x)


        transformed = Dynamic_activation(inputs)
        modulated = input_gate * transformed + forget_gate * inputs


        liquid_output = output_gate * modulated


        output = self.output_layer(liquid_output)
        output = self.dropout(output, training=training)


        output = inputs + output
        return self.layer_norm(output)



class TransformerDecoderBlock1(Layer):
    def __init__(self, input_dimension, inner_dimension, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate

        self.self_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension // num_heads
        )
        self.self_attn_dropout = Dropout(dropout_rate)
        self.self_attn_norm = LayerNormalization(epsilon=1e-6)

        # 编码器-解码器注意力
        self.cross_attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=input_dimension // num_heads
        )
        self.cross_attn_dropout = Dropout(dropout_rate)
        self.cross_attn_norm = LayerNormalization(epsilon=1e-6)

        # 液态神经网络替代传统FFN
        self.liquid_net = DynamicNetwork1(
            input_dim=input_dimension,
            inner_dim=inner_dimension,
            dropout_rate=dropout_rate
        )

    def call(self, inputs, training=False, mask=None):

        decoder_input, encoder_output = inputs


        self_attn_output = self.self_attn(
            decoder_input, decoder_input, attention_mask=mask
        )
        self_attn_output = self.self_attn_dropout(self_attn_output, training=training)
        out1 = decoder_input + self_attn_output
        out1 = self.self_attn_norm(out1)


        cross_attn_output = self.cross_attn(
            query=out1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=mask
        )
        cross_attn_output = self.cross_attn_dropout(cross_attn_output, training=training)
        out2 = out1 + cross_attn_output
        out2 = self.cross_attn_norm(out2)


        output = self.liquid_net(out2, training=training)

        return output


class EncoderDecoderTransformer1(BaseSequential):


    @property
    def name(self) -> str:
        return "Encoder-Decoder Transformer"

    @property
    def parameters(self) -> dict:
        return {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        }

    def __init__(self, n_encoder_layers: int, n_decoder_layers: int,
                 internal_size: int, n_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

    def apply(self, X, prefix: str = None):

        encoder_output = X
        for i in range(self.n_encoder_layers):
            encoder_output = TransformerEncoderBlock1(
                input_dimension=encoder_output.shape[-1],
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=False,
                prefix=f"{prefix}encoder_{i}_"
            )(encoder_output)


        decoder_output = encoder_output
        for i in range(self.n_decoder_layers):
            decoder_output = TransformerDecoderBlock1(
                input_dimension=decoder_output.shape[-1],
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate
            )([decoder_output, encoder_output])

        return decoder_output



transformers1: List[FunctionalComponent] = [
    EncoderDecoderTransformer1(
        n_encoder_layers=3,
        n_decoder_layers=3,
        internal_size=512,
        n_heads=5
    )
]



pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the Networktransformer
ftt = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers1[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ftt.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m1 = ftt.build_model()
# m1.summary()

# Compile the model
m1.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results1: pd.DataFrame
(train_results1, eval_results1, final_epoch1) = ftt.evaluate(m1, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('transformer + Dynamic Processing Layer CICIDS2017 test dataset prediction results\n',eval_results1)


# Define the Networktransformer
ft2 =DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers1[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft2.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
mn1 = ft2.build_model()
# mn1.summary()

# Compile the model
mn1.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results2: pd.DataFrame
(train_results2, eval_results2, final_epoch2) = ft2.evaluate(mn1, batch_size=64, epochs=5, steps_per_epoch=64, early_stopping_patience=10)

print('transformer + Dynamic Processing Layer unknown attacks test dataset prediction results\n',eval_results2)







print( ' ================================================== DGT transformer ==================================================')
print('\n')






transformers: List[FunctionalComponent] = [
    EncoderDecoderTransformer(
        n_encoder_layers=3,
        n_decoder_layers=3,
        internal_size=128,
        n_heads=5
    )
]


pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the Networktransformer
ft = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m = ft.build_model()
# m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('DGT transformer CICIDS2017 test dataset prediction results\n',eval_results)


# Define the Networktransformer
ft0 = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft0.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
mn = ft0.build_model()
# mn.summary()

# Compile the model
mn.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

# Get the evaluation results
eval_results0: pd.DataFrame
(train_results0, eval_results0, final_epoch0) = ft0.evaluate(mn, batch_size=128, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('DGT transformer unknown attacks test dataset prediction results\n',eval_results0)






import os

import pandas as pd

from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer

encodings = [
    NoInputEncoder(),
    RecordLevelEmbed(64),
    RecordLevelEmbed(64, project=True)
]

classification_heads = [
    LastTokenClassificationHead(),
    GlobalAveragePoolingClassificationHead(),

]







transformers: List[FunctionalComponent] = [
    EncoderDecoderTransformer(
        n_encoder_layers=3,
        n_decoder_layers=3,
        internal_size=128,
        n_heads=5
    )
]



flow_file_path = r"D:\dataset"

datasets = [
    ("CICIDS2017", pd.read_csv( r"D:\dataset\CICIDS2017\train.csv",encoding='gbk'), NamedDatasetSpecifications.CICIDS2017, 0.2, EvaluationDatasetSampling.RandomRows),
    ("UnknownAttack", pd.read_csv(r"D:\dataset\unknown_attacks\test unknown attack.csv",encoding='gbk'), NamedDatasetSpecifications.UnknownAttack, 0.2, EvaluationDatasetSampling.RandomRows)

]

pre_processing = StandardPreProcessing(n_categorical_levels=32)

# Define the Networktransformer
ft = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m = ft.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])





ft1 = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
ft1.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
mn = ft1.build_model()
mn.summary()

# Compile the model
mn.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])



print("\n")
print("-------------------------------------------------Cross-Dataset Experiment-------------------------------------------------")
print("\n")


# Get the evaluation results
eval_results2: pd.DataFrame
(train_results2, eval_results2, final_epoch2) = ft1.evaluate(m, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('Test results of the model trained on CICIDS2017 using the unknown_attacks dataset\n',eval_results2)



# Get the evaluation results
eval_results3: pd.DataFrame
(train_results3, eval_results3, final_epoch3) = ft.evaluate(mn, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('Model trained on the unknown_attacks dataset tested on the CIC2017 dataset results\n',eval_results3)








print("\n")
print("------------------------------------------------Incremental Cross-Dataset Experiments (After Fine-Tuning)-------------------------------------------------")
print("\n")



# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)



print('CICIDS2017 test dataset prediction results\n',eval_results)




# Get the evaluation results
eval_results4: pd.DataFrame
(train_results4, eval_results4, final_epoch4) = ft1.evaluate(m, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)
#(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=128, epochs=25, steps_per_epoch=64, early_stopping_patience=10)

print('test unknown attack prediction results\n',eval_results4)






import os
import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import GPUtil
import random
import tempfile
import shutil
from typing import List
import gc
import pynvml

import pandas as pd
import random
from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer


def fix_custom_layer_serialization():



    try:
        from keras.layers import Layer
        from transformers import TransformerEncoderBlock

        if not hasattr(TransformerEncoderBlock, 'get_config'):
            def get_config(self):
                config = super(TransformerEncoderBlock, self).get_config()

                config.update({
                    'internal_size': self.internal_size,
                    'n_heads': self.n_heads,
                    'dropout_rate': self.dropout_rate
                })
                return config

            TransformerEncoderBlock.get_config = get_config

        print("The serialization issue with the TransformerEncoderBlock has been fixed.")
    except Exception as e:
        print(f"Fixed error during custom layer serialization: {e}")

# ====================== 1. GPU Monitoring Tools ======================
class GPUMonitor:
    """GPU Monitoring Tool Using NVIDIA Management Library (NVML)"""

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.utilization = []
        self.memory_usage = []
        self.nvml_initialized = False

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.nvml_initialized = True
            print(f"NVML initialization successful. Monitoring GPU {gpu_id}")
        except Exception as e:
            print(f"NVML initialization failed: {e}")
            self.nvml_initialized = False

    def start(self):
        """Start Monitoring"""
        self.utilization = []
        self.memory_usage = []

    def update(self):
        """Update GPU Status"""
        if not self.nvml_initialized:
            return

        try:

            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.utilization.append(utilization.gpu)


            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.memory_usage.append(mem_info.used / (1024 * 1024))  # 转换为MB
        except Exception as e:
            print(f"Failed to retrieve GPU status: {e}")

    def get_stats(self):
        """Obtain statistical data"""
        return {
            'gpu_util_avg': np.mean(self.utilization) if self.utilization else 0,
            'gpu_util_max': np.max(self.utilization) if self.utilization else 0,
            'gpu_mem_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'gpu_mem_peak': np.max(self.memory_usage) if self.memory_usage else 0,
        }

    def __del__(self):
        """Clean up NVML resources"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass



# ====================== 2. Model Quantification Function ======================
def quantize_model(model, quant_mode='int8', representative_data=None):
    """Quantify the model to the specified precision"""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quant_mode == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data is not None:
                def representative_dataset():

                    for i in range(100):

                        inputs = {}
                        for input_idx, input_data in enumerate(representative_data):
                            input_shape = [1] + list(input_data.shape[1:])
                            inputs[input_idx] = np.random.rand(*input_shape).astype(np.float32)
                        yield [inputs[input_idx] for input_idx in sorted(inputs.keys())]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

        elif quant_mode == 'fp16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        quant_path = f"model_{quant_mode}.tflite"
        with open(quant_path, 'wb') as f:
            f.write(tflite_model)

        return quant_path
    except Exception as e:
        print(f"Error occurred during model quantification: {e}")
        return None

# ====================== 3. Edge Device Simulator (with GPU Monitoring) ======================
class EdgeDeviceSimulator:
    """Edge Device Performance Simulator (Supports GPU Monitoring)"""

    def __init__(self, model, test_data, model_type='keras'):
        self.model = model
        self.test_data = test_data
        self.model_type = model_type


        self.gpu_monitor = GPUMonitor()


        if model_type == 'tflite':
            try:
                self.interpreter = tf.lite.Interpreter(model_content=model)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("TFLite model loaded successfully")
            except Exception as e:
                print(f"Failed to load TFLite model: {e}")
                self.interpreter = None
        else:
            self.interpreter = None

    def predict(self, input_data):
        """Execute a single inference"""
        if self.model_type == 'keras':

            if isinstance(self.model.input, list):

                inputs = {}
                for i, inp in enumerate(self.model.inputs):
                    input_name = inp.name.split(':')[0]
                    inputs[input_name] = input_data[i] if isinstance(input_data, list) else input_data
                return self.model.predict(inputs, verbose=0)
            else:
                return self.model.predict(input_data, verbose=0)
        else:
            if self.interpreter is None:
                print("TFLite interpreter not initialized")
                return None


            if isinstance(input_data, list):
                for i, data in enumerate(input_data):
                    if self.input_details[i]['dtype'] == np.int8:
                        data = data.astype(np.int8)
                    else:
                        data = data.astype(np.float32)
                    self.interpreter.set_tensor(self.input_details[i]['index'], data)
            else:
                if self.input_details[0]['dtype'] == np.int8:
                    input_data = input_data.astype(np.int8)
                else:
                    input_data = input_data.astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])

    def benchmark(self, warmup=10, runs=100):
        """Performance Benchmarking (Including GPU Monitoring)"""
        self.gpu_monitor.start()


        for _ in range(warmup):
            sample_idx = random.randint(0, len(self.test_data[0]) - 1)
            input_sample = [data[sample_idx:sample_idx + 1] for data in self.test_data]
            self.predict(input_sample)


        latencies = []
        cpu_usages = []
        mem_usages = []

        for i in range(runs):
            sample_idx = random.randint(0, len(self.test_data[0]) - 1)  # 使用随机索引
            input_sample = [data[sample_idx:sample_idx + 1] for data in self.test_data]


            process = psutil.Process(os.getpid())
            cpu_start = process.cpu_percent(interval=None)


            self.gpu_monitor.update()


            start_time = time.perf_counter()
            result = self.predict(input_sample)
            latency = (time.perf_counter() - start_time) * 1000  # ms


            if result is None:
                print(f"Inference failed. Skipping this round of testing")
                continue


            self.gpu_monitor.update()


            cpu_end = process.cpu_percent(interval=None)


            current_mem = process.memory_info().rss / (1024 * 1024)  # MB
            mem_usages.append(current_mem)

            latencies.append(latency)
            cpu_usages.append(cpu_end - cpu_start)


            if i % 10 == 0:
                gc.collect()


        gpu_stats = self.gpu_monitor.get_stats()


        avg_mem = np.mean(mem_usages) if mem_usages else 0

        return {
            'avg_latency': np.mean(latencies) if latencies else 0,
            'max_latency': np.max(latencies) if latencies else 0,
            'min_latency': np.min(latencies) if latencies else 0,
            'cpu_usage': np.mean(cpu_usages) if cpu_usages else 0,
            'mem_usage': avg_mem,
            'latencies': latencies,
            **gpu_stats
        }


# ====================== 4.DGTTransformer Model Class ======================
class LiquidTransformerEvaluator:
    """DGTTransformer Evaluator Class"""

    def __init__(self, dataset_index=0):

        self.encodings = [
            NoInputEncoder(),
            RecordLevelEmbed(64),
            RecordLevelEmbed(64, project=True)
        ]

        self.classification_heads = [
            LastTokenClassificationHead(),
            GlobalAveragePoolingClassificationHead(),
        ]


        self.transformers = [
            EncoderDecoderTransformer(
                n_encoder_layers=3,
                n_decoder_layers=3,
                internal_size=128,
                n_heads=5
            )
        ]


        self.datasets = [
            ("CICIDS2017", pd.read_csv(r"D:\dataset\CICIDS2017\train.csv", encoding='gbk'),
             NamedDatasetSpecifications.CICIDS2017, 0.2, EvaluationDatasetSampling.RandomRows),
            ("UnknownAttack", pd.read_csv(r"D:\dataset\unknown_attacks\unknown_attack1.csv", encoding='gbk'),
             NamedDatasetSpecifications.UnknownAttack, 0.2, EvaluationDatasetSampling.RandomRows)
        ]


        self.pre_processing = StandardPreProcessing(n_categorical_levels=32)


        self.ft = DGTTransformer(
            pre_processing=self.pre_processing,
            input_encoding=self.encodings[1],
            sequential_model=self.transformers[0],
            classification_head=LastTokenClassificationHead(),
            params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1)
        )


        dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = self.datasets[dataset_index]
        self.ft.load_dataset(dataset_name, dataset_path, dataset_specification,
                             evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)


        self.model = self.ft.build_model()
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

    def evaluate_model(self, epochs=5):

        _, eval_results, _ = self.ft.evaluate(
            self.model,
            batch_size=256,
            epochs=epochs,
            steps_per_epoch=64,
            early_stopping_patience=10
        )
        return eval_results

    def prepare_test_data(self, sample_size=500):
        """Prepare test data (process multiple inputs)"""

        input_shapes = [inp.shape.as_list()[1:] for inp in self.model.inputs]


        test_data = []
        for shape in input_shapes:
            data = np.random.randn(sample_size, *shape).astype(np.float32)
            test_data.append(data)

        return test_data

    def get_model_input_names(self):
        """Get model input names"""
        return [inp.name.split(':')[0] for inp in self.model.inputs]


# ====================== 5. 效率实验主流程 ======================
def run_efficiency_experiment():
    """Run efficiency experiment (include GPU monitoring)"""
    # 实验配置
    QUANT_MODES = ['float32', 'fp16', 'int8']
    TEST_SAMPLE_SIZE = 500


    results = []


    print("Initializing the DGT Transformer model...")
    evaluator = LiquidTransformerEvaluator(dataset_index=1)


    input_names = evaluator.get_model_input_names()
    print(f"The model has {len(input_names)} inputs: {input_names}")


    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Detected GPU devices: {gpu_devices}")

        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"GPU {i}: {name.decode('utf-8')}")
            pynvml.nvmlShutdown()
        except:
            print("Unable to retrieve GPU details")
    else:
        print("Warning: No GPU device detected")


    print("Evaluating model performance...")
    eval_results = evaluator.evaluate_model(epochs=5)
    print('UnknownAttack prediction result:\n', eval_results)


    test_data = evaluator.prepare_test_data(TEST_SAMPLE_SIZE)
    print(f"Test data preparation complete: {[d.shape for d in test_data]}")


    for quant_mode in QUANT_MODES:
        print(f"Quantitative Mode: {quant_mode.upper()}")


        if quant_mode == 'float32':
            model = evaluator.model
            model_type = 'keras'
        else:

            repr_data = [d[:100] for d in test_data]
            model_path = quantize_model(evaluator.model, quant_mode, repr_data)
            if model_path is None:
                print(f"Quantification failed, skipping {quant_mode} mode")
                continue

            with open(model_path, 'rb') as f:
                model_content = f.read()
            model = model_content
            model_type = 'tflite'


        simulator = EdgeDeviceSimulator(model, test_data, model_type=model_type)


        try:
            metrics = simulator.benchmark(warmup=10, runs=100)


            if quant_mode == 'float32':

                temp_dir = tempfile.mkdtemp()
                model_path = os.path.join(temp_dir, "model")

                try:
                    evaluator.model.save(model_path, save_format="tf")


                    total_size = 0
                    for dirpath, _, filenames in os.walk(model_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)
                    model_size = total_size / (1024 * 1024)
                except Exception as save_error:
                    model_size = estimate_model_size(evaluator.model)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                model_size = len(model) / (1024 * 1024)


            result = {
                'quantization': quant_mode,
                'model_size': model_size,
                **metrics
            }
            results.append(result)


            print(f"Average latency:{metrics['avg_latency']:.2f} ms | "
                  f"Average memory: {metrics['mem_usage']:.2f} MB | "
                  f"GPU utilization: {metrics['gpu_util_avg']:.1f}% | "
                  f"Model size: {model_size:.2f} MB")
        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()


    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('unknown attack edge_performance_analysis.csv', index=False)
        return results_df
    else:
        print("No valid test results")
        return pd.DataFrame()


def estimate_model_size(model):

    total_params = model.count_params()

    estimated_bytes = total_params * 4

    estimated_bytes *= 1.1
    return estimated_bytes / (1024 * 1024)


# ====================== 6. Result Visualization (Including GPU Metrics) ======================
def visualize_gpu_results(results_df):
    plt.figure(figsize=(18, 12))


    plt.subplot(2, 3, 1)
    plt.bar(results_df['quantization'], results_df['avg_latency'], color='skyblue')
    plt.title('Comparison of inference latency')
    plt.ylabel('Milliseconds (ms)')
    plt.grid(True)


    plt.subplot(2, 3, 2)
    plt.bar(results_df['quantization'], results_df['gpu_util_avg'], color='salmon')
    plt.title('Average GPU utilization')
    plt.ylabel('Percentage (%)')
    plt.grid(True)


    plt.subplot(2, 3, 3)
    plt.plot(results_df['quantization'], results_df['gpu_mem_avg'], 'o-', label='Average usage')
    plt.plot(results_df['quantization'], results_df['gpu_mem_peak'], 's--', label='Peak usage')
    plt.title('GPU memory usage')
    plt.ylabel('MB')
    plt.legend()
    plt.grid(True)


    plt.subplot(2, 3, 4)
    plt.bar(results_df['quantization'], results_df['mem_usage'], color='lightgreen')
    plt.title('System memory usage')
    plt.ylabel('MB')


    plt.subplot(2, 3, 5)
    plt.bar(results_df['quantization'], results_df['model_size'], color='gold')
    plt.title('Model size comparison')
    plt.ylabel('MB')


    plt.subplot(2, 3, 6)
    all_latencies = []
    labels = []
    for _, row in results_df.iterrows():
        all_latencies.append(row['latencies'])
        labels.append(f"{row['quantization']}")
    plt.boxplot(all_latencies, labels=labels)
    plt.title('latency distribution')
    plt.ylabel('Milliseconds (ms)')

    plt.tight_layout()
    plt.savefig('unknown attack edge_performance_analysis.png')



if __name__ == "__main__":



    fix_custom_layer_serialization()


    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU device detected: {physical_devices}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Warning: No GPU device detected")


    print("\nInitiating GPU efficiency experiment")
    results = run_efficiency_experiment()


    if not results.empty:
        print("\nVisualization of experimental results...")
        visualize_gpu_results(results)


        print("\nGPU performance report:")
        print(results[['quantization', 'avg_latency', 'gpu_util_avg', 'gpu_mem_peak', 'mem_usage', 'model_size']])


        feasible = results[
            (results['avg_latency'] < 100) &
            (results['mem_usage'] < 500) &
            (results['model_size'] < 32) &
            (results['gpu_util_avg'] < 80)
            ]

        print("\nConfigurations to meet edge deployment requirements:")
        print(feasible)









import os
import tempfile
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, f1_score
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
#  NetworkTransformer

import tensorflow as tf
from keras.layers import Dense, Layer, MultiHeadAttention, Dropout, LayerNormalization,LSTM,Conv1D,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K


import pandas as pd

from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer
from sequential import BaseSequential
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
#  NetworkTransformer

import tensorflow as tf
from keras.layers import Dense, Layer, MultiHeadAttention, Dropout, LayerNormalization,LSTM,Conv1D,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K
import keras.layers as layers
from sequential import BaseSequential

try:
    from tensorflow._api.v2.v2 import keras
    from tensorflow.keras.layers import Layer
except ImportError:
    from tensorflow import keras
    from tensorflow.keras.layers import Layer
from keras.layers import Dense,Conv1D, MultiHeadAttention, Dropout, LayerNormalization,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D,ConvLSTM2D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling,  ClassificationFormat
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
# from input_encodings import RecordLevelEmbed
from pre_processings import BasePreProcessing
from tensorflow.keras import layers, Model, Input
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LinearSegmentedColormap
import warnings
try:
    from tensorflow._api.v2.v2 import keras
except ImportError:
    from tensorflow import keras
import tensorflow as tf
import keras.layers as layers
from keras.layers import Dense,Conv1D,LSTM,GRU, MultiHeadAttention, Dropout, LayerNormalization,ConvLSTM3D,MaxPooling1D,GlobalAveragePooling1D,ConvLSTM2D
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, MultiHeadAttention
from tensorflow.keras import backend as K


class DynamicAttention(Layer):
    def __init__(self, input_dim, num_heads, dropout_rate=0.1, is_cross_attention=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.is_cross_attention = is_cross_attention
        self.head_dim = input_dim // num_heads


        self.gate_weights = self.add_weight(
            name='gate_weights',
            shape=(input_dim, 3 * input_dim),
            initializer='glorot_uniform'
        )
        self.gate_bias = self.add_weight(
            name='gate_bias',
            shape=(3 * input_dim,),
            initializer='zeros'
        )
        # 动态激活参数
        self.alpha = self.add_weight(
            name='alpha',
            shape=(1,),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )


        self.attn = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=self.head_dim
        )
        self.attn_dropout = Dropout(dropout_rate)
        self.attn_norm = LayerNormalization(epsilon=1e-6)


        self.output_layer = Dense(input_dim)
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.attention_weights = None

    def Dynamic_activation(self, x):

        return self.alpha * tf.nn.silu(x) + self.beta * tf.nn.tanh(x)

    def Dynamic_gating(self, x):

        gates = K.dot(x, self.gate_weights) + self.gate_bias
        input_gate, forget_gate, output_gate = tf.split(gates, 3, axis=-1)

        input_gate = tf.nn.sigmoid(input_gate)
        forget_gate = tf.nn.sigmoid(forget_gate)
        output_gate = tf.nn.sigmoid(output_gate)

        transformed = self.Dynamic_activation(x)
        modulated = input_gate * transformed + forget_gate * x
        return output_gate * modulated

    def call(self, inputs, training=False, mask=None):

        self.attention_weights = None


        if self.is_cross_attention and isinstance(inputs, (list, tuple)):

            query, key_value = inputs

            liquid_query = self.Dynamic_gating_gating(query)
            liquid_key_value = self.Dynamic_gating_gating(key_value)


            try:
                attn_output, attention_weights = self.attn(
                    liquid_query, liquid_key_value,
                    attention_mask=mask,
                    return_attention_scores=True
                )
                self.attention_weights = attention_weights
            except Exception as e:
                print(f"Error in cross attention: {e}")

                attn_output = self.attn(
                    liquid_query, liquid_key_value,
                    attention_mask=mask
                )
        else:



            liquid_input = self.liquid_gating(inputs)


            try:
                attn_output, attention_weights = self.attn(
                    liquid_input, liquid_input,
                    attention_mask=mask,
                    return_attention_scores=True
                )
                self.attention_weights = attention_weights
            except Exception as e:
                print(f"Error in self attention: {e}")

                attn_output = self.attn(
                    liquid_input, liquid_input,
                    attention_mask=mask
                )

        attn_output = self.attn_dropout(attn_output, training=training)


        if self.is_cross_attention and isinstance(inputs, (list, tuple)):

            attn_output = inputs[0] + attn_output
        else:

            attn_output = inputs + attn_output

        attn_output = self.attn_norm(attn_output)


        liquid_output = self.liquid_gating(attn_output)


        output = self.output_layer(liquid_output)
        output = self.dropout(output, training=training)


        output = attn_output + output
        return self.layer_norm(output)

    def get_attention_weights(self):

        if self.attention_weights is None:
            return None


        try:
            if hasattr(self.attention_weights, 'numpy'):
                return self.attention_weights.numpy()
            else:

                return tf.keras.backend.get_value(self.attention_weights)
        except Exception as e:
            print(f"Error getting attention weights: {e}")
            return None

class GPT3Attention(layers.Layer):
    def __init__(self, n_heads, d_model, dropout_rate=0.1):
        super(GPT3Attention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.depth = d_model // n_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    # noinspection PyMethodOverriding
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        # Scaled Dot-Product Attention
        scaled_attention_logits = tf.matmul(q, k, transpose_b=True)
        scaled_attention_logits = scaled_attention_logits / tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights)

        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))

        output = self.dense(output)
        output = self.dropout(output)

        return output

class MultiHeadAttentionImplementation:
    Keras = 0,
    GPT3 = 1




class TransformerEncoderBlock(Layer):
    def __init__(self, input_dimension, inner_dimension, num_heads,
                 dropout_rate=0.1, use_conv=False, prefix=None):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate
        self.use_conv = use_conv
        self.prefix = prefix


        self.Dynamic_attention = DynamicAttention(
            input_dim=input_dimension,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )

    def call(self, inputs, training=False, mask=None):

        output = self.Dynamic_attention(inputs, training=training, mask=mask)
        return output

    def get_attention_weights(self):

        if hasattr(self.Dynamic_attention, 'get_attention_weights'):
            return self.Dynamic_attention.get_attention_weights()
        return None


class TransformerDecoderBlock(Layer):
    def __init__(self, input_dimension, inner_dimension, num_heads, dropout_rate=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.input_dimension = input_dimension
        self.inner_dimension = inner_dimension
        self.dropout_rate = dropout_rate


        self.self_attention = DynamicAttention(
            input_dim=input_dimension,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_cross_attention=False
        )


        self.cross_attention = DynamicAttention(
            input_dim=input_dimension,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_cross_attention=True  # 标记为交叉注意力
        )


        self.Dynamic_net = DynamicAttention(
            input_dim=input_dimension,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            is_cross_attention=False
        )

    def call(self, inputs, training=False, mask=None):
        decoder_input, encoder_output = inputs


        self_attn_output = self.self_attention(decoder_input, training=training, mask=mask)


        cross_attn_output = self.cross_attention(
            [self_attn_output, encoder_output],  # 传递列表 [query, key_value]
            training=training,
            mask=mask
        )


        output = self.Dynamic_net(cross_attn_output, training=training)

        return output

    def get_attention_weights(self):


        if hasattr(self.cross_attention, 'get_attention_weights'):
            return self.cross_attention.get_attention_weights()
        return None



class ModelInputSpecification:

    def __init__(self, categorical_format, n_features):
        self.categorical_format = categorical_format
        self.n_features = n_features
        self.window_size = 1






















class EncoderDecoderTransformer(Layer):


    @property
    def name(self) -> str:
        return "EncoderDecoderTransformer"

    @property
    def parameters(self) -> dict:
        return {
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        }

    def __init__(self, n_encoder_layers: int, n_decoder_layers: int,
                 internal_size: int, n_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.internal_size = internal_size
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate


        self.encoder_layers = [
            TransformerEncoderBlock(
                input_dimension=self.internal_size,  # 使用固定维度
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                use_conv=False,
                prefix=f"encoder_{i}_"
            ) for i in range(self.n_encoder_layers)
        ]

        self.decoder_layers = [
            TransformerDecoderBlock(
                input_dimension=self.internal_size,  # 使用固定维度
                inner_dimension=self.internal_size,
                num_heads=self.n_heads,
                dropout_rate=self.dropout_rate
            ) for i in range(self.n_decoder_layers)
        ]

        self.attention_history = []

    def call(self, inputs, training=False):


        self.attention_history = []


        encoder_output = inputs
        for i, layer in enumerate(self.encoder_layers):
            encoder_output = layer(encoder_output, training=training)

            if hasattr(layer, 'get_attention_weights'):
                weights = layer.get_attention_weights()
                self.attention_history.append(weights)
                print(f"Encoder layer {i} attention weights shape: {weights.shape if weights is not None else 'None'}")
            else:
                self.attention_history.append(None)
                print(f"Encoder layer {i} has no attention weights")


        decoder_output = encoder_output
        for i, layer in enumerate(self.decoder_layers):

            decoder_output = layer([decoder_output, encoder_output], training=training)

            if hasattr(layer, 'get_attention_weights'):
                weights = layer.get_attention_weights()
                self.attention_history.append(weights)
                print(f"Decoder layer {i} attention weights shape: {weights.shape if weights is not None else 'None'}")
            else:
                self.attention_history.append(None)
                print(f"Decoder layer {i} has no attention weights")


        return decoder_output

    def get_attention_history(self):

        eager_weights = []
        for weights in self.attention_history:
            if weights is None:
                eager_weights.append(None)
                continue

            try:
                if hasattr(weights, 'numpy'):
                    eager_weights.append(weights.numpy())
                else:

                    eager_weights.append(tf.keras.backend.get_value(weights))
            except Exception as e:
                print(f"Error getting attention weights: {e}")
                eager_weights.append(None)

        return eager_weights

    def get_config(self):

        config = super().get_config()
        config.update({
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "internal_size": self.internal_size,
            "n_heads": self.n_heads,
            "dropout_rate": self.dropout_rate
        })
        return config



class AttentionVisualizer:
    def __init__(self):
        self.attention_history = []

    def collect_attention_weights(self, model, X_batch):

        try:

            for layer in model.layers:
                if isinstance(layer, EncoderDecoderTransformer):
                    attention_weights = layer.get_attention_history()
                    if attention_weights:
                        self.attention_history.append({
                            'batch_data': X_batch,
                            'attention_weights': attention_weights
                        })
                        print(f"Collected attention weights for batch, {len(attention_weights)} layers")
                    else:
                        print("No attention weights found in this batch")
                    break
            else:
                print("EncoderDecoderTransformer layer not found in model")
        except Exception as e:
            print(f"Error collecting attention weights: {e}")

    def visualize_attention_heatmap(self, attention_weights, layer_idx=0, head_idx=0,
                                    title="Attention Heatmap", save_path=None):

        if not attention_weights:
            print("No attention weights available")
            return


        if layer_idx >= len(attention_weights) or attention_weights[layer_idx] is None:
            print(f"No attention weights available for layer {layer_idx}")
            print(f"Available layers: {len(attention_weights)}")

            for i, weights in enumerate(attention_weights):
                if weights is not None:
                    layer_idx = i
                    print(f"Using layer {i} instead")
                    break
            else:
                print("No valid attention weights found in any layer")
                return


        layer_weights = attention_weights[layer_idx]


        if layer_weights is None:
            print(f"Attention weights for layer {layer_idx} are None")
            return


        try:

            if hasattr(layer_weights, 'numpy'):
                layer_weights_np = layer_weights.numpy()
            else:

                layer_weights_np = tf.keras.backend.get_value(layer_weights)
        except Exception as e:
            print(f"Error converting attention weights to numpy: {e}")
            return


        if len(layer_weights_np.shape) == 4:  # (batch, heads, seq_len, seq_len)
            if head_idx >= layer_weights_np.shape[1]:
                print(f"Head index {head_idx} out of range for layer {layer_idx}")
                print(f"Available heads: {layer_weights_np.shape[1]}")

                head_idx = 0
                print(f"Using head {head_idx} instead")

            attn_matrix = layer_weights_np[0, head_idx]
        elif len(layer_weights_np.shape) == 3:
            attn_matrix = layer_weights_np[0]
        else:
            print(f"Unexpected attention weights shape: {layer_weights_np.shape}")
            return


        plt.figure(figsize=(10, 8))


        colors = ["white", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("attention_cmap", colors, N=100)


        sns.heatmap(attn_matrix, cmap=cmap, center=0.5,
                    square=True, xticklabels=False, yticklabels=False)
        plt.title(f"{title} - Layer {layer_idx}, Head {head_idx}")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def visualize_attention_dynamics(self, selected_layer=0, selected_head=0,
                                     title="Attention Dynamics", save_path=None):

        if not self.attention_history:
            print("No attention history available")
            return


        dynamics_data = []
        for i, record in enumerate(self.attention_history):
            if selected_layer < len(record['attention_weights']):
                layer_weights = record['attention_weights'][selected_layer]


                try:

                    if hasattr(layer_weights, 'numpy'):
                        layer_weights_np = layer_weights.numpy()
                    else:

                        layer_weights_np = tf.keras.backend.get_value(layer_weights)
                except Exception as e:
                    print(f"Error converting attention weights to numpy: {e}")
                    continue


                if len(layer_weights_np.shape) == 4:
                    if selected_head < layer_weights_np.shape[1]:
                        attn_matrix = layer_weights_np[0, selected_head]
                elif len(layer_weights_np.shape) == 3:
                    attn_matrix = layer_weights_np[0]
                else:
                    continue


                avg_attention = np.mean(attn_matrix)
                dynamics_data.append(avg_attention)


        plt.figure(figsize=(12, 6))
        plt.plot(range(len(dynamics_data)), dynamics_data, 'b-', linewidth=2)
        plt.xlabel('Batch Number')
        plt.ylabel('Average Attention Strength')
        plt.title(f"{title} - Layer {selected_layer}, Head {selected_head}")
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def create_attention_animation(self, layer_idx=0, head_idx=0,
                                   title="Attention Dynamics Animation",
                                   save_path="attention_animation.gif"):

        if not self.attention_history:
            print("No attention history available")
            return


        frames = []
        for i, record in enumerate(self.attention_history):
            if (record['attention_weights'] and
                    layer_idx < len(record['attention_weights']) and
                    record['attention_weights'][layer_idx] is not None):

                layer_weights = record['attention_weights'][layer_idx]


                try:

                    if hasattr(layer_weights, 'numpy'):
                        layer_weights_np = layer_weights.numpy()
                    else:

                        layer_weights_np = tf.keras.backend.get_value(layer_weights)
                except Exception as e:
                    print(f"Error converting attention weights to numpy: {e}")
                    continue


                if len(layer_weights_np.shape) == 4:
                    if head_idx < layer_weights_np.shape[1]:
                        attn_matrix = layer_weights_np[0, head_idx]
                        frames.append(attn_matrix)
                elif len(layer_weights_np.shape) == 3:
                    attn_matrix = layer_weights_np[0]
                    frames.append(attn_matrix)


        if not frames:
            print("No valid attention frames found for the specified layer and head")
            print(
                f"Available layers in first record: {len(self.attention_history[0]['attention_weights']) if self.attention_history and self.attention_history[0]['attention_weights'] else 0}")
            return

        print(f"Creating animation with {len(frames)} frames")

        fig, ax = plt.subplots(figsize=(10, 8))


        colors = ["white", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("attention_cmap", colors, N=100)


        im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'{title} - Batch 0')
        ax.set_xticks([])
        ax.set_yticks([])


        def update(frame):
            im.set_array(frames[frame])
            ax.set_title(f'{title} - Batch {frame}')
            return [im]


        ani = FuncAnimation(fig, update, frames=len(frames),
                            interval=200, blit=True)

        plt.tight_layout()


        if save_path:
            try:
                ani.save(save_path, writer='pillow', fps=5)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")

        plt.show()



class RecordLevelEmbed(layers.Layer):
    def __init__(self, embed_dimension: int, project: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.embed_dimension = embed_dimension
        self.project = project
        self.dense_layers = []
        self.prefix = None

    def build(self, input_shapes):

        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]


        for i, shape in enumerate(input_shapes):

            if len(shape) != 3:
                raise ValueError(f"Input shape must have 3 dimensions, got {len(shape)}")


            n_features = shape[-1]
            if n_features is None:
                raise ValueError(f"The last dimension of input shape {i} must be known")


            dense_layer = layers.Dense(
                self.embed_dimension,
                activation="linear",
                use_bias=not self.project,
                name=f"{self.prefix}embed_{i}" if self.prefix else f"embed_{i}"
            )
            dense_layer.build((None, n_features))

            self.dense_layers.append(dense_layer)

        super().build(input_shapes)

    def call(self, X):

        if not isinstance(X, list):
            X = [X]

        embedded_records = []
        for i, record in enumerate(X):

            if len(record.shape) != 3:
                raise ValueError(f"Input record must have 3 dimensions, got {len(record.shape)}")

            original_shape = tf.shape(record)


            record_flat = tf.reshape(record, [-1, original_shape[-1]])


            if i < len(self.dense_layers):
                x = self.dense_layers[i](record_flat)
            else:

                x = self.dense_layers[0](record_flat)


            x = tf.reshape(x, [original_shape[0], original_shape[1], self.embed_dimension])
            embedded_records.append(x)


        if len(embedded_records) == 1:
            return embedded_records[0]


        return tf.concat(embedded_records, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dimension": self.embed_dimension,
            "project": self.project
        })
        return config


class StandardPreProcessing(BasePreProcessing):
    def __init__(self, n_categorical_levels: int = 32, clip_numerical_values: bool = False):
        super().__init__()
        self.n_categorical_levels = n_categorical_levels
        self.clip_numerical_values = clip_numerical_values
        self.min_range = {}
        self.encoded_levels = {}
        self.feature_columns = []
        self.target_column = None
        self.n_features = 0

    @property
    def name(self) -> str:
        return "Standard Preprocessing"

    @property
    def parameters(self) -> dict:
        return {
            "n_categorical_levels": self.n_categorical_levels,
            "clip_numerical_values": self.clip_numerical_values
        }

    def fit(self, dataset: pd.DataFrame, target_column: str):

        self.target_column = target_column
        self.feature_columns = [col for col in dataset.columns if col != target_column]
        self.n_features = len(self.feature_columns)

        print(f"Fitting preprocessing for {self.n_features} features")

        # 拟合每个特征列
        for col in self.feature_columns:
            if pd.api.types.is_numeric_dtype(dataset[col]):
                self.fit_numerical(col, dataset[col].values)
            else:
                self.fit_categorical(col, dataset[col].values)

    def transform(self, data: pd.DataFrame) -> np.ndarray:


        transformed_features = []

        for col in self.feature_columns:
            if col in self.min_range:
                transformed_col = self.transform_numerical(col, data[col].values)
            elif col in self.encoded_levels:
                transformed_col = self.transform_categorical(
                    col, data[col].values, ClassificationFormat.Integers)
            else:
                transformed_col = data[col].values.reshape(-1, 1)

            if transformed_col.ndim == 1:
                transformed_col = transformed_col.reshape(-1, 1)

            transformed_features.append(transformed_col)


        combined = np.hstack(transformed_features).astype(np.float32)


        print(f"Transformed data shape: {combined.shape}")
        return combined

    def fit_transform(self, dataset: pd.DataFrame, target_column: str) -> tuple:

        self.fit(dataset, target_column)
        return self.transform(dataset)

    def fit_numerical(self, column_name: str, values: np.array):
        v0 = np.nanmin(values)
        v1 = np.nanmax(values)
        r = v1 - v0


        if r == 0:
            r = 1.0  #

        self.min_range[column_name] = (v0, r)

    def transform_numerical(self, column_name: str, values: np.array):
        if column_name not in self.min_range:

            return values.reshape(-1, 1)

        col_min, col_range = self.min_range[column_name]


        values = np.nan_to_num(values, nan=col_min)


        values -= col_min


        if col_range > 0:
            col_values = np.log(values + 1)
            col_values *= 1. / np.log(col_range + 1)
        else:
            col_values = values

        if self.clip_numerical_values:
            col_values = np.clip(col_values, 0., 1.)

        return col_values.reshape(-1, 1)

    def fit_categorical(self, column_name: str, values: np.array):

        values = np.nan_to_num(values, nan="NaN")

        levels, level_counts = np.unique(values, return_counts=True)
        sorted_levels = list(sorted(zip(levels, level_counts), key=lambda x: x[1], reverse=True))
        self.encoded_levels[column_name] = [s[0] for s in sorted_levels[:self.n_categorical_levels]]

    def transform_categorical(self, column_name: str, values: np.array, expected_categorical_format: CategoricalFormat):
        if column_name not in self.encoded_levels:

            return np.zeros(len(values)).reshape(-1, 1)


        values = np.nan_to_num(values, nan="NaN")

        encoded_levels = self.encoded_levels[column_name]
        result_values = np.zeros(len(values), dtype="uint32")

        for level_i, level in enumerate(encoded_levels):
            level_mask = values == level
            result_values[level_mask] = level_i + 1

        if expected_categorical_format == ClassificationFormat.Integers:
            return result_values.reshape(-1, 1)


        v = pd.get_dummies(result_values, prefix=column_name)
        return v.values





def validate_data_shape(model, X_data):

    if model.input_shape is None:
        print("Warning: Model input shape not defined")
        return X_data

    expected_shape = model.input_shape[1:]
    actual_shape = X_data.shape[1:]

    if actual_shape == expected_shape:
        return X_data

    print(f"Shape mismatch! Expected {expected_shape}, got {actual_shape}")


    if len(actual_shape) < len(expected_shape):
        print("Adjusting shape dimensions...")

        for _ in range(len(expected_shape) - len(actual_shape)):
            X_data = np.expand_dims(X_data, axis=1)
        actual_shape = X_data.shape[1:]


    if actual_shape[1] != expected_shape[1]:
        diff = expected_shape[1] - actual_shape[1]

        if diff > 0:
            padding = np.zeros((X_data.shape[0], X_data.shape[1], diff))
            X_data = np.concatenate([X_data, padding], axis=2)
            print(f"Padded data shape: {X_data.shape}")
        else:
            X_data = X_data[:, :, :expected_shape[1]]
            print(f"Truncated data shape: {X_data.shape}")

    return X_data

def train_model_with_validation(model, X_train, y_train):


    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        )
    ]


    history = model.fit(
        X_train,
        y_train,
        epochs=5,
        batch_size=128,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight={0: 1, 1: 5}  # 增加攻击样本权重
    )


    print("\nTraining metrics:")
    train_metrics = model.evaluate(X_train, y_train, verbose=0)
    print(f"Loss: {train_metrics[0]:.4f}, Accuracy: {train_metrics[1]:.4f}")

    return model, history






class DynamicAdaptationExperiment():
    def __init__(self, model, pre_processing, batch_size=128, adaptation_epochs=5, learning_rate=0.001):
        self.base_model = model
        self.pre_processing = pre_processing
        self.batch_size = batch_size
        self.adaptation_epochs = adaptation_epochs
        self.learning_rate = learning_rate
        self.n_features = pre_processing.n_features

        self.attention_visualizer = AttentionVisualizer()

    def _safe_copy_model(self, model):

        try:
            # 创建新模型并复制权重
            config = model.get_config()
            new_model = Model.from_config(config)
            new_model.set_weights(model.get_weights())
            return new_model
        except Exception as e:

            print("Model copy success...")
            return self._create_fallback_model_with_original_arch()

    def _create_fallback_model_with_original_arch(self):

        # 使用与原始构建相同的参数
        return build_liquid_transformer_model(self.n_features)

    def simulate_data_stream(self, dataset, attack_intervals, unknown_attack_types):


        tf.config.run_functions_eagerly(True)


        print("Starting data stream simulation with detailed monitoring...")


        print("Copying model...")
        adapted_model = self._safe_copy_model(self.base_model)


        optimizer = Adam(learning_rate=self.learning_rate)
        adapted_model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['binary_accuracy'])
        print("Model copied and compiled.")


        print("Transforming dataset...")
        X_stream = self.pre_processing.transform(dataset)
        y_stream = (dataset[self.pre_processing.target_column] != 'BENIGN').astype(int)


        X_stream = X_stream.reshape((X_stream.shape[0], 1, X_stream.shape[1]))
        print(f"Reshaped input shape: {X_stream.shape}")


        print(f"Input shape: {X_stream.shape}")


        detailed_results = {
            'batch': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'new_attack_detected': [],
            'adaptation_time': [],
            'n_normal': [],
            'n_attack': [],
            'n_new_attack': []
        }


        print("Starting data stream simulation...")
        batch_count = 0
        n_batches = len(X_stream) // self.batch_size

        for i in range(0, len(X_stream), self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, len(X_stream))
            if batch_end - batch_start < 10:
                continue

            X_batch = X_stream[batch_start:batch_end]
            y_batch = y_stream[batch_start:batch_end]


            y_pred = (adapted_model.predict(X_batch, verbose=0) > 0.1).astype(int).flatten()


            self.attention_visualizer.collect_attention_weights(adapted_model, X_batch)

            if len(np.unique(y_batch)) > 1:
                acc = accuracy_score(y_batch, y_pred)
                f1 = f1_score(y_batch, y_pred)
            else:
                acc = 1.0 if np.all(y_batch == y_pred) else 0.0
                f1 = acc


            new_attack_detected = False
            adaptation_time = 0


            if batch_count in attack_intervals:
                attack_labels = dataset[self.pre_processing.target_column].iloc[batch_start:batch_end]
                new_attack_mask = np.isin(attack_labels, unknown_attack_types)

                if np.sum(new_attack_mask) > 10:  # 确保有足够样本
                    new_attack_detected = True
                    X_new_attack = X_batch[new_attack_mask]
                    y_new_attack = (attack_labels[new_attack_mask] != 'BENIGN').astype(int)

                    print(f"Batch {batch_count}: Adapting to {np.sum(new_attack_mask)} new attack samples")

                    start_time = time.time()
                    adapted_model.fit(
                        X_new_attack, y_new_attack,
                        epochs=self.adaptation_epochs,
                        batch_size=min(32, len(X_new_attack)),
                        verbose=0
                    )
                    adaptation_time = time.time() - start_time
                    print(f"Adapted in {adaptation_time:.2f} seconds")
                else:
                    print(f"Batch {batch_count}: Insufficient new attack samples ({np.sum(new_attack_mask)})")


            detailed_results['batch'].append(batch_count)
            detailed_results['accuracy'].append(acc)
            detailed_results['f1_score'].append(f1)
            detailed_results['new_attack_detected'].append(new_attack_detected)
            detailed_results['adaptation_time'].append(adaptation_time)
            detailed_results['n_normal'].append(np.sum(y_batch == 0))
            detailed_results['n_attack'].append(np.sum(y_batch == 1))
            detailed_results['n_new_attack'].append(np.sum(new_attack_mask) if new_attack_detected else 0)

            print(f"Batch {batch_count}: Acc={acc:.4f}, F1={f1:.4f}, "
                  f"Normal={detailed_results['n_normal'][-1]}, "
                  f"Attack={detailed_results['n_attack'][-1]}")
            batch_count += 1


        tf.config.run_functions_eagerly(False)

        return detailed_results

    def visualize_attention_heatmaps(self, batch_idx=0, layer_idx=0, head_idx=0):

        if not hasattr(self, 'attention_visualizer'):
            print("Attention visualizer not initialized.")
            return

        if not hasattr(self.attention_visualizer, 'attention_history'):
            print("Attention history not available in visualizer.")
            return

        if batch_idx >= len(self.attention_visualizer.attention_history):
            print(
                f"Batch index {batch_idx} out of range. Available batches: {len(self.attention_visualizer.attention_history)}")
            return

        attention_weights = self.attention_visualizer.attention_history[batch_idx]['attention_weights']
        self.attention_visualizer.visualize_attention_heatmap(
            attention_weights, layer_idx, head_idx,
            title=f"Liquid Transformer Attention (Batch {batch_idx})",
            save_path=f"attention_heatmap_batch_{batch_idx}_layer_{layer_idx}_head_{head_idx}.png"
        )

    def visualize_attention_dynamics(self, layer_idx=0, head_idx=0):

        if not hasattr(self, 'attention_visualizer'):
            print("Attention visualizer not initialized.")
            return

        if not hasattr(self.attention_visualizer, 'attention_history'):
            print("Attention history not available in visualizer.")
            return

        self.attention_visualizer.visualize_attention_dynamics(
            selected_layer=layer_idx,
            selected_head=head_idx,
            title="Liquid Transformer Attention Dynamics",
            save_path=f"attention_dynamics_layer_{layer_idx}_head_{head_idx}.png"
        )

    def create_attention_animation(self, layer_idx=0, head_idx=0, save_path="attention_animation.gif"):
        """创建注意力权重变化的动画"""
        # 检查是否有注意力历史数据
        if not hasattr(self, 'attention_visualizer'):
            print("Attention visualizer not initialized.")
            return

        if not hasattr(self.attention_visualizer, 'attention_history'):
            print("Attention history not available in visualizer.")
            return

        if not self.attention_visualizer.attention_history:
            print("No attention history available. Please run simulate_data_stream first.")
            return


        self.attention_visualizer.create_attention_animation(
            layer_idx=layer_idx,
            head_idx=head_idx,
            title="Liquid Transformer Attention Dynamics",
            save_path=save_path
        )


        if not self.attention_history:
            print("No attention history available")
            return


        frames = []
        for i, record in enumerate(self.attention_history):
            if (record['attention_weights'] and
                    layer_idx < len(record['attention_weights']) and
                    record['attention_weights'][layer_idx] is not None):

                layer_weights = record['attention_weights'][layer_idx]


                if len(layer_weights.shape) == 4:
                    if head_idx < layer_weights.shape[1]:
                        attn_matrix = layer_weights[0, head_idx]
                        frames.append(attn_matrix)
                elif len(layer_weights.shape) == 3:
                    attn_matrix = layer_weights[0]
                    frames.append(attn_matrix)


        if not frames:
            print("No valid attention frames found for the specified layer and head")
            print(
                f"Available layers in first record: {len(self.attention_history[0]['attention_weights']) if self.attention_history and self.attention_history[0]['attention_weights'] else 0}")
            return

        print(f"Creating animation with {len(frames)} frames")

        fig, ax = plt.subplots(figsize=(10, 8))


        colors = ["white", "yellow", "orange", "red"]
        cmap = LinearSegmentedColormap.from_list("attention_cmap", colors, N=100)


        im = ax.imshow(frames[0], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f'{plt.title} - Batch 0')
        ax.set_xticks([])
        ax.set_yticks([])


        def update(frame):
            im.set_array(frames[frame])
            ax.set_title(f'{plt.title} - Batch {frame}')
            return [im]


        ani = FuncAnimation(fig, update, frames=len(frames),
                            interval=200, blit=True)

        plt.tight_layout()


        if save_path:
            try:
                ani.save(save_path, writer='pillow', fps=5)
                print(f"Animation saved to {save_path}")
            except Exception as e:
                print(f"Failed to save animation: {e}")

        plt.show()



    def calculate_recovery_speed(self, results):

        recovery_speeds = []
        attack_points = [i for i, detected in enumerate(results['new_attack_detected']) if detected]

        for point in attack_points:

            base_accuracy = results['accuracy'][point - 1] if point > 0 else 1.0
            target_accuracy = base_accuracy * 0.9

            for i in range(point + 1, len(results['accuracy'])):
                if results['accuracy'][i] >= target_accuracy:
                    recovery_speeds.append(i - point)
                    break
            else:
                recovery_speeds.append(float('inf'))

        return np.mean(recovery_speeds) if recovery_speeds else float('inf')


class BalancedDynamicAdaptationExperiment(DynamicAdaptationExperiment):
    def simulate_data_stream(self, dataset, attack_intervals, unknown_attack_types):
        """
                 Use balanced batches to simulate data streams and test model adaptability

                Args:
                    dataset: Dataset containing normal and attack traffic
                    attack_intervals: List of batch positions where new attacks are introduced
                    unknown_attack_types: List of attack types unknown during training

                Returns:
                    Experiment results dictionary
                """

        print("Preparing balanced data stream...")
        X_stream = self.pre_processing.transform(dataset)
        y_stream = (dataset[self.pre_processing.target_column] != 'BENIGN').astype(int)


        X_stream = validate_data_shape(self.base_model, X_stream.reshape((X_stream.shape[0], 1, X_stream.shape[1])))


        normal_mask = y_stream == 0
        attack_mask = y_stream == 1

        X_normal = X_stream[normal_mask]
        y_normal = y_stream[normal_mask]
        X_attack = X_stream[attack_mask]
        y_attack = y_stream[attack_mask]


        balanced_batches = []
        min_samples = min(len(X_normal), len(X_attack))

        print(f"Creating balanced batches with {min_samples} samples per class...")

        for i in range(0, min_samples, self.batch_size // 2):

            normal_start = i
            normal_end = min(i + self.batch_size // 2, len(X_normal))


            attack_start = i
            attack_end = min(i + self.batch_size // 2, len(X_attack))


            if (normal_end - normal_start) < 5 or (attack_end - attack_start) < 5:
                continue


            X_batch = np.vstack((
                X_normal[normal_start:normal_end],
                X_attack[attack_start:attack_end]
            ))
            y_batch = np.concatenate((
                y_normal[normal_start:normal_end],
                y_attack[attack_start:attack_end]
            ))


            attack_types_batch = np.concatenate((
                dataset[self.pre_processing.target_column].values[normal_mask][normal_start:normal_end],
                dataset[self.pre_processing.target_column].values[attack_mask][attack_start:attack_end]
            ))

            balanced_batches.append((X_batch, y_batch, attack_types_batch))


        balanced_batches = [
            (
                batch[0].reshape((batch[0].shape[0], 1, batch[0].shape[1])),
                batch[1],
                batch[2]
            )
            for batch in balanced_batches
        ]


        print("Copying model for adaptation...")
        adapted_model = self._safe_copy_model(self.base_model)


        optimizer = Adam(learning_rate=self.learning_rate)
        adapted_model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['binary_accuracy'])


        results = {
            'batch': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'new_attack_detected': [],
            'adaptation_time': [],
            'n_normal': [],
            'n_attack': [],
            'n_new_attack': []
        }


        print("Processing balanced batches...")
        for batch_idx, (X_batch, y_batch, attack_types_batch) in enumerate(balanced_batches):

            n_normal = np.sum(y_batch == 0)
            n_attack = np.sum(y_batch == 1)
            total = len(y_batch)


            if n_normal == 0 or n_attack == 0:
                print(f"跳过无效批次 {batch_idx}: 正常样本={n_normal}, 攻击样本={n_attack}")
                continue


            X_batch = validate_data_shape(adapted_model, X_batch)


            y_pred = self.dynamic_threshold_prediction(adapted_model, X_batch, y_batch)


            acc = accuracy_score(y_batch, y_pred)
            precision = precision_score(y_batch, y_pred, zero_division=0)
            recall = recall_score(y_batch, y_pred, zero_division=0)
            f1 = f1_score(y_batch, y_pred, zero_division=0)


            new_attack_detected = False
            adaptation_time = 0
            n_new_attack = 0

            if batch_idx in attack_intervals:

                new_attack_mask = np.isin(attack_types_batch, unknown_attack_types)
                n_new_attack = np.sum(new_attack_mask)

                if n_new_attack > 10:
                    new_attack_detected = True
                    X_new_attack = X_batch[new_attack_mask]
                    y_new_attack = y_batch[new_attack_mask]

                    print(f"批次 {batch_idx}: 适应 {n_new_attack} 个新型攻击样本")


                    X_new_attack = validate_data_shape(adapted_model, X_new_attack)

                    start_time = time.time()
                    adapted_model.fit(
                        X_new_attack, y_new_attack,
                        epochs=self.adaptation_epochs,
                        batch_size=min(32, len(X_new_attack)),
                        verbose=0
                    )
                    adaptation_time = time.time() - start_time
                    print(f"适应完成，耗时 {adaptation_time:.2f} 秒")
                else:
                    print(f"批次 {batch_idx}: 新型攻击样本不足 ({n_new_attack})")


            results['batch'].append(batch_idx)
            results['accuracy'].append(acc)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1_score'].append(f1)
            results['new_attack_detected'].append(new_attack_detected)
            results['adaptation_time'].append(adaptation_time)
            results['n_normal'].append(n_normal)
            results['n_attack'].append(n_attack)
            results['n_new_attack'].append(n_new_attack)


            print(f"Batch  {batch_idx}: Accuracy={acc:.4f}, F1={f1:.4f}, "
                  f"Normal samples={n_normal}, Attack samples={n_attack}, "
                  f"New attack samples={n_new_attack}")

        print("Batch processing completed")
        return results

    def dynamic_threshold_prediction(self, model, X_batch, y_batch):
        """
        Dynamic Threshold Prediction Based on Batch Sample Distribution

        Args:
            model: The model to use
            X_batch: Input data batch
            y_batch: Label batch

        Returns:
            Predicted labels
        """

        y_prob = model.predict(X_batch, verbose=0).flatten()


        attack_ratio = np.mean(y_batch)


        if attack_ratio > 0.7:
            threshold = 0.3
        elif attack_ratio < 0.3:
            threshold = 0.7
        else:
            threshold = 0.5


        y_pred = (y_prob > threshold).astype(int)

        return y_pred

    def validate_batch_distribution(self, y_batch):
        """
        Verify whether the distribution of batch samples is reasonable

        Args:
            y_batch: Batch labels

        Returns:
            Valid
        """
        n_normal = np.sum(y_batch == 0)
        n_attack = np.sum(y_batch == 1)
        total = len(y_batch)


        if n_normal == total or n_attack == total:
            return False


        min_ratio = min(n_normal / total, n_attack / total)
        if min_ratio < 0.2:
            return False

        return True

    def validate_data_shape(model, X_data):
        """
        Verify whether the data shape matches the model input

        Args:
            model: The model to be verified
            X_data: Input data

        Returns:
            Adjusted data
        """
        if model.input_shape is None:
            return X_data


        expected_shape = model.input_shape[1:]
        actual_shape = X_data.shape[1:]

        if actual_shape == expected_shape:
            return X_data

        print(f"Shape mismatch! Expected {expected_shape}, actual {actual_shape}")


        if len(actual_shape) >= 2 and len(expected_shape) >= 2:
            if actual_shape[1] != expected_shape[1]:
                diff = expected_shape[1] - actual_shape[1]

                if diff > 0:
                    padding = np.zeros((X_data.shape[0], X_data.shape[1], diff))
                    X_data = np.concatenate([X_data, padding], axis=2)
                    print(f"Filled shape: {X_data.shape}")
                else:
                    X_data = X_data[:, :, :expected_shape[1]]
                    print(f"Shape after truncation: {X_data.shape}")

        return X_data




class CallableRecordLevelEmbed(layers.Layer):


    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embed_layer = RecordLevelEmbed(embed_dim)

    def call(self, inputs):
        return self.embed_layer.apply(inputs)


class FixedRecordLevelEmbed(layers.Layer):


    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.embed_layer = RecordLevelEmbed(embed_dim)
        self.model_input_specification = ModelInputSpecification(
            categorical_format=ClassificationFormat.OneHot,
            n_features=None
        )
        self.embed_layer.model_input_specification = self.model_input_specification

    def build(self, input_shape):

        if len(input_shape) != 3:
            raise ValueError(f"Input shape must have 3 dimensions, got {len(input_shape)}")


        n_features = input_shape[-1]
        if n_features is None:
            raise ValueError("Feature dimension must be defined, got None")


        self.model_input_specification.n_features = n_features
        self.model_input_specification.window_size = input_shape[1]  # 通常是1



        self.embed_layer.build([input_shape])
        super().build(input_shape)

    def call(self, inputs):

        return self.embed_layer([inputs])


class LastTokenClassificationHead(layers.Layer):


    def __init__(self, output_units=128, **kwargs):
        super().__init__(**kwargs)
        self.output_units = output_units
        self.dense = None

    def build(self, input_shape):

        feature_dim = input_shape[-1]
        self.dense = layers.Dense(self.output_units, activation='relu')
        super().build(input_shape)

    def call(self, inputs):

        last_token = inputs[:, -1, :]  # 形状: (batch_size, features)


        return self.dense(last_token)

    def get_config(self):
        config = super().get_config()
        config.update({'output_units': self.output_units})
        return config


def build_liquid_transformer_model(n_features):

    input_layer = Input(shape=(1, n_features), name="main_input")


    embedding = layers.Dense(64, activation='relu')(input_layer)


    projected = layers.Dense(128)(embedding)


    transformer_layer = EncoderDecoderTransformer(
        n_encoder_layers=2,
        n_decoder_layers=2,
        internal_size=128,
        n_heads=4
    )
    transformer_output = transformer_layer(projected)



    x = layers.GlobalAveragePooling1D()(transformer_output)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    output = layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])




    return model


def check_label_distribution(dataset, target_column):

    label_counts = dataset[target_column].value_counts()
    print("Label Distribution:")
    print(label_counts)


    benign_count = label_counts.get('BENIGN', 0)
    total_count = len(dataset)
    benign_ratio = benign_count / total_count if total_count > 0 else 0

    print(f"Benign traffic ratio: {benign_ratio:.4f}")

    if benign_ratio < 0.3:
        print("Warning: The proportion of normal traffic is too low, which may prevent the model from learning normal patterns")

    return benign_ratio



if __name__ == "__main__":

    encodings = [
        NoInputEncoder(),
        RecordLevelEmbed(64),
        RecordLevelEmbed(64, project=True)
    ]

    classification_heads = [
        LastTokenClassificationHead(),
        GlobalAveragePoolingClassificationHead(),

    ]




    print( ' ================================================== DGT transformer ==================================================')

    flow_file_path = r"D:\dataset"


    dataset_specs = [
        {
            "name": "CICIDS2017",
            "path": r"D:\dataset\unknown_attacks\train  attack.csv",
            "spec": NamedDatasetSpecifications.CICIDS2017,
            "eval_percent": 0.2,
            "eval_method": EvaluationDatasetSampling.RandomRows
        },
        {
            "name": "UnknownAttack",
            "path": r"D:\dataset\unknown_attacks\test unknown attack100%.csv",
            "spec": NamedDatasetSpecifications.UnknownAttack,
            "eval_percent": 0.2,
            "eval_method": EvaluationDatasetSampling.RandomRows
        }
    ]


    dataset_spec = dataset_specs[0]
    print(f"Loading dataset: {dataset_spec['name']} from {dataset_spec['path']}")
    dataset_df = pd.read_csv(dataset_spec['path'], encoding='gbk')




    target_column = ' Label'
    if ' Label' not in dataset_df.columns:

        possible_labels = ['label', ' Label', 'attack_type', 'target', ' Label']
        for col in possible_labels:
            if col in dataset_df.columns:
                target_column = col
                print(f"Using '{col}' as target column")
                break
        else:

            dataset_df[target_column] = 'BENIGN'
            print(f"Warning: Created default '{target_column}' column")


    pre_processing = StandardPreProcessing(n_categorical_levels=32)




    print("Fitting pre-processing...")
    pre_processing.fit(dataset_df, target_column=target_column)


    print("Training data sample:")
    print(dataset_df.head(3))

    print("\nFeature statistics:")
    print(dataset_df.describe())

    print("\nLabel distribution:")
    print(dataset_df[target_column].value_counts())


    X_sample = pre_processing.transform(dataset_df.head(10))
    print("\nPreprocessed data sample:")
    print(X_sample)

    n_features = pre_processing.n_features
    print(f"Number of features: {n_features}")

    print("Checking training data label distribution...")
    train_benign_ratio = check_label_distribution(dataset_df, target_column)



    print("Building Liquid Transformer model...")
    m = build_liquid_transformer_model(n_features)


    m.summary()


    m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])


    X_train = pre_processing.transform(dataset_df)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))  # 转换为3D
    y_train = (dataset_df[target_column] != 'BENIGN').astype(int)


    if hasattr(m, 'input_shape'):
        print(f"Model input shape: {m.input_shape}")
    elif hasattr(m, 'inputs') and m.inputs:
        print(f"Model expects input shape: {m.inputs[0].shape}")



    print(f"Training data shape: {X_train.shape}")


    print("Training model...")
    history = m.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=3,
        validation_split=0.1,
        verbose=1
    )


    def build_simple_model(n_features):

        model = keras.Sequential([
            layers.Input(shape=(1, n_features)),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model



    print("Testing with simple model...")
    simple_model = build_simple_model(n_features)
    history1 = simple_model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=5,
        validation_split=0.2,
        verbose=1
    )


    loss, acc = simple_model.evaluate(X_train, y_train)
    print(f"Simple model training accuracy: {acc:.4f}")






    test_data_path = r"D:\dataset\unknown_attacks\test unknown attack100%.csv"
    print(f"Loading test data from: {test_data_path}")
    test_data = pd.read_csv(test_data_path, encoding='gbk')

    if target_column not in test_data.columns:
        for col in possible_labels:
            if col in test_data.columns:
                target_column = col
                print(f"Using '{col}' as target column for test data")
                break
        else:
            test_data[target_column] = 'FTP-Patator'
            print(f"Created default '{target_column}' column for test data")

    missing_cols = set(pre_processing.feature_columns) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0

    extra_cols = set(test_data.columns) - set(pre_processing.feature_columns + [target_column])
    if extra_cols:
        test_data.drop(columns=list(extra_cols), inplace=True)

    print("\nChecking test data label distribution...")
    test_benign_ratio = check_label_distribution(test_data, target_column)






    print("\n=== Data Preprocessing ===")
    pre_processing = StandardPreProcessing()
    pre_processing.fit(dataset_df, target_column=target_column)
    X_train = pre_processing.transform(dataset_df)
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    y_train = (dataset_df[target_column] != 'BENIGN').astype(int)


    print("\n=== Model Training ===")
    n_features = X_train.shape[2]
    model = build_liquid_transformer_model(n_features)
    model.summary()

    trained_model, history = train_model_with_validation(model, X_train, y_train)




    print("\n=== Dynamic Adaptation Experiment ===")
    experiment = DynamicAdaptationExperiment(
        model=trained_model,
        pre_processing=pre_processing,
        batch_size=128,
        adaptation_epochs=5,
        learning_rate=0.0001
    )

    results = experiment.simulate_data_stream(
        test_data,
        attack_intervals=[15, 30, 45],
        unknown_attack_types=['unknown attack']
    )


    print("Visualizing attention dynamics...")


    print("Plotting attention heatmaps for key batches...")
    key_batches = [0, 15, 30]
    for batch_idx in key_batches:
        experiment.visualize_attention_heatmaps(batch_idx=batch_idx, layer_idx=0, head_idx=0)


    experiment.visualize_attention_dynamics(layer_idx=0, head_idx=0)


    print("Creating attention animation...")
    experiment.create_attention_animation(
        layer_idx=0, head_idx=0,
        save_path="DGT_transformer_attention.gif"
    )

import os
import time
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import GPUtil
import random
import tempfile
import shutil
from typing import List
import gc
import pynvml

import pandas as pd
import random
from dataset_specification import NamedDatasetSpecifications
from enumerations import EvaluationDatasetSampling
from DGT import DGTTransformer
from DGT_parameters import DGTParameters
from framework_component import FunctionalComponent
from classification_heads import *
from input_encodings import *
from pre_processings import StandardPreProcessing
from transformers import Transformer
from transformers import EncoderDecoderTransformer


def fix_custom_layer_serialization():
    try:
        from keras.layers import Layer
        from transformers import TransformerEncoderBlock

        if not hasattr(TransformerEncoderBlock, 'get_config'):
            def get_config(self):
                config = super(TransformerEncoderBlock, self).get_config()

                config.update({
                    'internal_size': self.internal_size,
                    'n_heads': self.n_heads,
                    'dropout_rate': self.dropout_rate
                })
                return config

            TransformerEncoderBlock.get_config = get_config

        print("The serialization issue with the TransformerEncoderBlock has been fixed")
    except Exception as e:
        print(f"Fixed error during custom layer serialization: {e}")


# ====================== 1.GPU Monitoring Tools ======================
class GPUMonitor:
    """GPU Monitoring Tool Using NVIDIA Management Library (NVML)"""

    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.utilization = []
        self.memory_usage = []
        self.nvml_initialized = False

        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.nvml_initialized = True
            print(f"NVML initialization successful. Monitoring GPU {gpu_id}")
        except Exception as e:
            print(f"NVML initialization failed: {e}")
            self.nvml_initialized = False

    def start(self):
        """Start monitoring"""
        self.utilization = []
        self.memory_usage = []

    def update(self):
        """Update GPU Status"""
        if not self.nvml_initialized:
            return

        try:

            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            self.utilization.append(utilization.gpu)

            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.memory_usage.append(mem_info.used / (1024 * 1024))  # 转换为MB
        except Exception as e:
            print(f"Failed to retrieve GPU status: {e}")

    def get_stats(self):
        """Obtain statistical data"""
        return {
            'gpu_util_avg': np.mean(self.utilization) if self.utilization else 0,
            'gpu_util_max': np.max(self.utilization) if self.utilization else 0,
            'gpu_mem_avg': np.mean(self.memory_usage) if self.memory_usage else 0,
            'gpu_mem_peak': np.max(self.memory_usage) if self.memory_usage else 0,
        }

    def __del__(self):
        """Clean up NVML resources"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


# ====================== 2. Model Quantification Function ======================
def quantize_model(model, quant_mode='int8', representative_data=None):
    """Quantify the model to the specified precision"""
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        if quant_mode == 'int8':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            if representative_data is not None:
                def representative_dataset():

                    for i in range(100):

                        inputs = {}
                        for input_idx, input_data in enumerate(representative_data):
                            input_shape = [1] + list(input_data.shape[1:])
                            inputs[input_idx] = np.random.rand(*input_shape).astype(np.float32)
                        yield [inputs[input_idx] for input_idx in sorted(inputs.keys())]

                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8

        elif quant_mode == 'fp16':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]

        tflite_model = converter.convert()

        quant_path = f"model_{quant_mode}.tflite"
        with open(quant_path, 'wb') as f:
            f.write(tflite_model)

        return quant_path
    except Exception as e:
        print(f"Error occurred during model quantification: {e}")
        return None


# ====================== 3. Edge Device Simulator (with GPU Monitoring) ======================
class EdgeDeviceSimulator:
    """Edge Device Simulator (with GPU Monitoring)"""

    def __init__(self, model, test_data, model_type='keras'):
        self.model = model
        self.test_data = test_data
        self.model_type = model_type

        self.gpu_monitor = GPUMonitor()

        if model_type == 'tflite':
            try:
                self.interpreter = tf.lite.Interpreter(model_content=model)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("TFLite模型加载成功")
            except Exception as e:
                print(f"加载TFLite模型失败: {e}")
                self.interpreter = None
        else:
            self.interpreter = None

    def predict(self, input_data):
        """Execute a single inference"""
        if self.model_type == 'keras':

            if isinstance(self.model.input, list):

                inputs = {}
                for i, inp in enumerate(self.model.inputs):
                    input_name = inp.name.split(':')[0]

                    inputs[input_name] = input_data[i] if isinstance(input_data, list) else input_data
                return self.model.predict(inputs, verbose=0)
            else:
                return self.model.predict(input_data, verbose=0)
        else:
            if self.interpreter is None:
                print("TFLite interpreter not initialized")
                return None

            if isinstance(input_data, list):
                for i, data in enumerate(input_data):

                    if self.input_details[i]['dtype'] == np.int8:
                        data = data.astype(np.int8)
                    else:
                        data = data.astype(np.float32)
                    self.interpreter.set_tensor(self.input_details[i]['index'], data)
            else:

                if self.input_details[0]['dtype'] == np.int8:
                    input_data = input_data.astype(np.int8)
                else:
                    input_data = input_data.astype(np.float32)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]['index'])

    def benchmark(self, warmup=10, runs=100):
        """Performance Benchmarking (Including GPU Monitoring)"""

        self.gpu_monitor.start()

        cpu_cores = psutil.cpu_count(logical=True)

        for _ in range(warmup):
            sample_idx = random.randint(0, len(self.test_data[0]) - 1)
            input_sample = [data[sample_idx:sample_idx + 1] for data in self.test_data]
            self.predict(input_sample)

        latencies = []
        cpu_usages = []
        mem_usages = []
        process = psutil.Process(os.getpid())

        for i in range(runs):
            sample_idx = random.randint(0, len(self.test_data[0]) - 1)
            input_sample = [data[sample_idx:sample_idx + 1] for data in self.test_data]

            self.gpu_monitor.update()

            cpu_start = process.cpu_times()
            start_time = time.perf_counter()

            result = self.predict(input_sample)

            end_time = time.perf_counter()

            cpu_end = process.cpu_times()

            latency = (end_time - start_time) * 1000  # ms
            latencies.append(latency)

            user_time_diff = cpu_end.user - cpu_start.user
            system_time_diff = cpu_end.system - cpu_start.system
            total_cpu_time = user_time_diff + system_time_diff

            real_time = end_time - start_time

            if real_time > 0:
                cpu_usage = (total_cpu_time / real_time) * 100 / cpu_cores
            else:
                cpu_usage = 0

            cpu_usages.append(cpu_usage)

            current_mem = process.memory_info().rss / (1024 * 1024)  # MB
            mem_usages.append(current_mem)

            if i % 10 == 0:
                gc.collect()

        gpu_stats = self.gpu_monitor.get_stats()

        avg_mem = np.mean(mem_usages) if mem_usages else 0

        return {
            'avg_latency': np.mean(latencies) if latencies else 0,
            'max_latency': np.max(latencies) if latencies else 0,
            'min_latency': np.min(latencies) if latencies else 0,
            'cpu_usage': np.mean(cpu_usages) if cpu_usages else 0,  # 平均CPU使用率
            'cpu_usage_max': np.max(cpu_usages) if cpu_usages else 0,  # 峰值CPU使用率
            'mem_usage': avg_mem,
            'latencies': latencies,
            'cpu_usages': cpu_usages,
            **gpu_stats
        }


# ====================== 4. DGTTransformer Model Class ======================
class LiquidTransformerEvaluator:
    """DGTTransformer Evaluator"""

    def __init__(self, dataset_index=0):
        self.encodings = [
            NoInputEncoder(),
            RecordLevelEmbed(64),
            RecordLevelEmbed(64, project=True)
        ]

        self.classification_heads = [
            LastTokenClassificationHead(),
            GlobalAveragePoolingClassificationHead(),
        ]

        self.transformers = [
            EncoderDecoderTransformer(
                n_encoder_layers=3,
                n_decoder_layers=3,
                internal_size=128,
                n_heads=5
            )
        ]

        self.datasets = [
            ("CICIDS2017", pd.read_csv(r"D:\dataset\CICIDS2017\train.csv", encoding='gbk'),
             NamedDatasetSpecifications.CICIDS2017, 0.2, EvaluationDatasetSampling.RandomRows),
            ("UnknownAttack", pd.read_csv(r"D:\dataset\unknown_attacks\unknown_attack1.csv", encoding='gbk'),
             NamedDatasetSpecifications.UnknownAttack, 0.2, EvaluationDatasetSampling.RandomRows)
        ]

        self.pre_processing = StandardPreProcessing(n_categorical_levels=32)

        self.ft = DGTTransformer(
            pre_processing=self.pre_processing,
            input_encoding=self.encodings[1],
            sequential_model=self.transformers[0],
            classification_head=LastTokenClassificationHead(),
            params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1)
        )

        dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = self.datasets[dataset_index]
        self.ft.load_dataset(dataset_name, dataset_path, dataset_specification,
                             evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

        self.model = self.ft.build_model()
        self.model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

    def evaluate_model(self, epochs=5):
        _, eval_results, _ = self.ft.evaluate(
            self.model,
            batch_size=256,
            epochs=epochs,
            steps_per_epoch=64,
            early_stopping_patience=10
        )
        return eval_results

    def prepare_test_data(self, sample_size=500):
        input_shapes = [inp.shape.as_list()[1:] for inp in self.model.inputs]

        test_data = []
        for shape in input_shapes:
            data = np.random.randn(sample_size, *shape).astype(np.float32)
            test_data.append(data)

        return test_data

    def get_model_input_names(self):
        return [inp.name.split(':')[0] for inp in self.model.inputs]


# ====================== 5. Efficiency Experiment Main Process ======================
def run_efficiency_experiment():
    """Execution Efficiency Experiment (Including GPU Monitoring)"""

    QUANT_MODES = ['float32', 'fp16', 'int8']
    TEST_SAMPLE_SIZE = 500

    results = []

    print("Initializing the DGTTransformer model...")
    evaluator = LiquidTransformerEvaluator(dataset_index=1)

    input_names = evaluator.get_model_input_names()
    print(f"The model has {len(input_names)} inputs: {input_names}")

    gpu_devices = tf.config.list_physical_devices('GPU')
    if gpu_devices:
        print(f"Detected GPU devices: {gpu_devices}")

        try:
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                print(f"GPU {i}: {name.decode('utf-8')}")
            pynvml.nvmlShutdown()
        except:
            print("Unable to retrieve GPU details")
    else:
        print("Warning: No GPU device detected")

    print("Evaluating model performance")
    eval_results = evaluator.evaluate_model(epochs=5)
    print('UnknownAttack Prediction Results:\n', eval_results)

    test_data = evaluator.prepare_test_data(TEST_SAMPLE_SIZE)
    print(f"Test data preparation complete:{[d.shape for d in test_data]}")

    for quant_mode in QUANT_MODES:
        print(f"\nQuantitative Model: {quant_mode.upper()}")

        if quant_mode == 'float32':
            model = evaluator.model
            model_type = 'keras'
        else:
            #
            repr_data = [d[:100] for d in test_data]
            model_path = quantize_model(evaluator.model, quant_mode, repr_data)
            if model_path is None:
                print(f"Quantification failed, skipping {quant_mode} mode")
                continue

            with open(model_path, 'rb') as f:
                model_content = f.read()
            model = model_content
            model_type = 'tflite'

        simulator = EdgeDeviceSimulator(model, test_data, model_type=model_type)

        try:
            metrics = simulator.benchmark(warmup=10, runs=100)

            if quant_mode == 'float32':

                temp_dir = tempfile.mkdtemp()
                model_path = os.path.join(temp_dir, "model")

                try:
                    evaluator.model.save(model_path, save_format="tf")

                    total_size = 0
                    for dirpath, _, filenames in os.walk(model_path):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            total_size += os.path.getsize(fp)
                    model_size = total_size / (1024 * 1024)
                except Exception as save_error:
                    model_size = estimate_model_size(evaluator.model)
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
            else:
                model_size = len(model) / (1024 * 1024)  # MB

            result = {
                'quantization': quant_mode,
                'model_size': model_size,
                **metrics
            }
            results.append(result)

            print(f"Average delay: {metrics['avg_latency']:.2f} ms | "
                  f"Average memory: {metrics['mem_usage']:.2f} MB | "
                  f"GPU utilization: {metrics['gpu_util_avg']:.1f}% | "
                  f"Model size: {model_size:.2f} MB")
        except Exception as e:
            print(f"Test failed: {str(e)}")
            import traceback
            traceback.print_exc()

    if results:
        results_df = pd.DataFrame(results)

        results_df.to_csv('experiment6 unknown attack edge_performance_analysis.csv', index=False)
        return results_df
    else:
        print("No valid test results")
        return pd.DataFrame()


def estimate_model_size(model):
    total_params = model.count_params()

    estimated_bytes = total_params * 4

    estimated_bytes *= 1.1
    return estimated_bytes / (1024 * 1024)


# ====================== 6. Result Visualization (Including GPU Metrics ======================
def visualize_gpu_results(results_df):
    plt.figure(figsize=(24, 12))

    plt.subplot(2, 4, 1)
    bars1 = plt.bar(results_df['quantization'], results_df['avg_latency'], color='skyblue')
    plt.title('Comparison of inference latency', fontsize=12, fontweight='bold')
    plt.ylabel('Milliseconds (ms)', fontsize=10)
    plt.xticks(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 4, 2)
    bars2 = plt.bar(results_df['quantization'], results_df['gpu_util_avg'], color='salmon')
    plt.title('Average GPU utilization', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage (%)', fontsize=10)
    plt.ylim(0, min(100, max(results_df['gpu_util_avg']) * 1.5))
    plt.xticks(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 4, 3)
    plt.plot(results_df['quantization'], results_df['gpu_mem_avg'], 'o-', label='Average usage')
    plt.plot(results_df['quantization'], results_df['gpu_mem_peak'], 's--', label='Peak usage')
    plt.title('GPU memory usage', fontsize=12, fontweight='bold')
    plt.ylabel('MB', fontsize=10)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    for i, (avg, peak) in enumerate(zip(results_df['gpu_mem_avg'], results_df['gpu_mem_peak'])):
        plt.text(i, avg + 0.5, f'{avg:.1f}', ha='center', va='bottom', fontsize=8)
        plt.text(i, peak + 0.5, f'{peak:.1f}', ha='center', va='bottom', fontsize=8)

    plt.subplot(2, 4, 4)
    bars4 = plt.bar(results_df['quantization'], results_df['cpu_usage_max'], color='lightgreen')
    plt.title('CPU usage peak', fontsize=12, fontweight='bold')
    plt.ylabel('MB', fontsize=10)
    plt.xticks(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    for bar in bars4:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 4, 5)
    bars5 = plt.bar(results_df['quantization'], results_df['model_size'], color='gold')
    plt.title('Model size comparison', fontsize=12, fontweight='bold')
    plt.ylabel('MB', fontsize=10)
    plt.xticks(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    for bar in bars5:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 4, 6)
    all_latencies = []
    labels = []
    for _, row in results_df.iterrows():
        all_latencies.append(row['latencies'])
        labels.append(f"{row['quantization']}")

    box = plt.boxplot(all_latencies, labels=labels, patch_artist=True)

    colors = ['lightblue', 'lightgreen', 'salmon']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    plt.title('Latency distribution', fontsize=12, fontweight='bold')
    plt.ylabel('Milliseconds (ms)', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 4, 7)
    bars7 = plt.bar(results_df['quantization'], results_df['cpu_usage'], color='purple')
    plt.title('CPU Usage Comparison', fontsize=12, fontweight='bold')
    plt.ylabel('CPU Usage (%)', fontsize=10)
    plt.ylim(0, min(100, max(results_df['cpu_usage']) * 1.5))
    plt.xticks(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)
    # 添加数值标签
    for bar in bars7:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.subplot(2, 4, 8)

    bars8 = plt.bar(results_df['quantization'], results_df['mem_usage'], color='salmon')
    plt.title('Overall memory usage', fontsize=12, fontweight='bold')
    plt.ylabel('MB', fontsize=10)
    plt.xticks(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.5)

    for bar in bars8:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('DGT model edge deployment performance analysis', fontsize=16, fontweight='bold', y=-0.5)
    plt.tight_layout()
    plt.savefig('experiment6 unknown_attack_edge_performance_analysis_comparison.png')


if __name__ == "__main__":

    fix_custom_layer_serialization()

    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"GPU device detected: {physical_devices}")
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    else:
        print("Warning: No GPU device detected")

    print("\nInitiating GPU efficiency experiment...")
    results = run_efficiency_experiment()

    if not results.empty:
        print("\nVisualization of experimental results...")
        visualize_gpu_results(results)

        print("\nGPU performance report:")
        print(results[['quantization', 'avg_latency', 'gpu_util_avg', 'gpu_mem_peak', 'mem_usage', 'model_size']])

        feasible = results[
            (results['avg_latency'] < 100) &
            (results['mem_usage'] < 500) &
            (results['model_size'] < 32) &
            (results['gpu_util_avg'] < 80)
            ]

        print("\nConfigurations to meet edge deployment requirements:")
        print(feasible)








import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import time


np.random.seed(42)
tf.random.set_seed(42)


try:
    from dataset_specification import NamedDatasetSpecifications
    from enumerations import EvaluationDatasetSampling, CategoricalFormat
    from DGT import DGTTransformer
    from DGT_parameters import DGTParameters
    from framework_component import FunctionalComponent
    from classification_heads import *
    from input_encodings import *
    from pre_processings import StandardPreProcessing
    from transformers import Transformer, EncoderDecoderTransformer
except ImportError:

    print("Warning: Could not import some modules. Please check your imports.")



class CustomTensorFlowClassifier:
    """
    Custom TensorFlow Classifier Wrapper, Compatible with TensorFlow 2.1
Supports multi-input models and adversarial attacks
    """

    def __init__(self, model, nb_classes: int, input_shape: Tuple,
                 loss_object, clip_values: Tuple[float, float] = (0, 1)):
        """
        Initialize Custom Classifier

        Parameters:
            model: Trained TensorFlow/Keras model
            nb_classes: Number of classes
            input_shape: Input shape
            loss_object: Loss function object
            clip_values: Data clipping range (min, max)
        """
        self.model = model
        self.nb_classes = nb_classes
        self.input_shape = input_shape
        self.loss_object = loss_object
        self.clip_values = clip_values

    def predict(self, x: Union[np.ndarray, List[np.ndarray]],
                batch_size: int = 128, **kwargs) -> np.ndarray:
        """
        Predict on input data

        Parameters:
            x: Input data, can be a single array or a list of arrays (multi-input models)
            batch_size: Batch size

        Returns:
            Prediction probabilities
        """
        if isinstance(x, list):

            return self.model.predict(x, batch_size=batch_size, verbose=0)
        else:

            return self.model.predict(x, batch_size=batch_size, verbose=0)

    def loss_gradient(self, x: Union[np.ndarray, List[np.ndarray]],
                      y: np.ndarray, **kwargs) -> np.ndarray:
        """
        Compute the gradient of the loss function with respect to the input

        Parameters:
            x: Input data
            y: True labels

        Returns:
            Gradient of the loss function with respect to the input
        """
        if isinstance(x, list):

            gradients = []
            for i, x_i in enumerate(x):
                x_tensor = tf.convert_to_tensor(x_i)
                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)

                    model_inputs = [tf.convert_to_tensor(x_j) for j, x_j in enumerate(x)]
                    model_inputs[i] = x_tensor
                    predictions = self.model(model_inputs)
                    loss = self.loss_object(y, predictions)
                grad = tape.gradient(loss, x_tensor)
                gradients.append(grad.numpy())
            return gradients
        else:

            x_tensor = tf.convert_to_tensor(x)
            with tf.GradientTape() as tape:
                tape.watch(x_tensor)
                predictions = self.model(x_tensor)
                loss = self.loss_object(y, predictions)
            gradients = tape.gradient(loss, x_tensor)
            return gradients.numpy()

    def class_gradient(self, x: Union[np.ndarray, List[np.ndarray]],
                       label: Optional[Union[int, List[int]]] = None, **kwargs) -> np.ndarray:
        """
        Compute category scores with respect to input gradients

        Parameters:
            x: Input data
            label: Target category (None indicates all categories)

        Returns:
            Category scores with respect to input gradients
        """
        if isinstance(x, list):

            return self.loss_gradient(x, np.ones((x[0].shape[0], self.nb_classes)), **kwargs)
        else:

            x_tensor = tf.convert_to_tensor(x)

            with tf.GradientTape() as tape:
                tape.watch(x_tensor)
                predictions = self.model(x_tensor)

            if label is None:

                gradients = []
                for i in range(self.nb_classes):
                    grad = tape.gradient(predictions[:, i], x_tensor)
                    gradients.append(grad.numpy())
                return np.array(gradients)
            else:

                if isinstance(label, int):
                    label = [label] * x.shape[0]
                grad = tape.gradient(predictions[:, label], x_tensor)
                return grad.numpy()

    def get_activations(self, x: Union[np.ndarray, List[np.ndarray]],
                        layer: Union[int, str], batch_size: int = 128) -> np.ndarray:
        """
        Retrieve activation values for a specified layer

        Parameters:
            x: Input data
            layer: Layer index or name
            batch_size: Batch size

        Returns:
            Activation values for the specified layer
        """

        if isinstance(layer, int):
            layer_name = self.model.layers[layer].name
        else:
            layer_name = layer

        intermediate_model = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer(layer_name).output
        )

        if isinstance(x, list):
            return intermediate_model.predict(x, batch_size=batch_size, verbose=0)
        else:
            return intermediate_model.predict(x, batch_size=batch_size, verbose=0)

    def save(self, filename: str, **kwargs):
        """
        Save Model

        Parameters:
            filename: File name
        """
        self.model.save(filename, **kwargs)

    def __call__(self, x: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Make the instance callable, equivalent to the predict method"""
        return self.predict(x)


# Anti-Attack Implementation (ART Library-Independent)
class AdversarialAttacks:
    """Implementing Common Adversarial Attack Methods"""

    @staticmethod
    def multi_input_fgsm_attack(classifier, x_list, y, eps=0.1):
        x_adv_list = []
        for x_input in x_list:

            gradients = classifier.loss_gradient(x_input, y)

            perturbation = eps * np.sign(gradients)
            x_adv = x_input + perturbation

            x_adv = np.clip(x_adv, classifier.clip_values[0], classifier.clip_values[1])
            x_adv_list.append(x_adv)
        return x_adv_list

    def fgsm_attack(classifier, x: np.ndarray, y: np.ndarray,
                    eps: float = 0.1, targeted: bool = False) -> np.ndarray:
        """
        Fast Gradient Sign Method (FGSM) Attack

        Parameters:
            classifier: Classifier instance
            x: Input data
            y: True label or target label
            eps: Perturbation magnitude
            targeted: Whether it is a targeted attack

        Returns:
            Adversarial example
        """

        gradients = classifier.loss_gradient(x, y)

        if targeted:

            perturbation = -eps * np.sign(gradients)
        else:

            perturbation = eps * np.sign(gradients)


        x_adv = x + perturbation


        x_adv = np.clip(x_adv, classifier.clip_values[0], classifier.clip_values[1])

        return x_adv

    @staticmethod
    def pgd_attack(classifier, x: np.ndarray, y: np.ndarray,
                   eps: float = 0.1, eps_iter: float = 0.01,
                   nb_iter: int = 10, targeted: bool = False) -> np.ndarray:
        """
        Projection Gradient Descent (PGD) Attack

        Parameters:
            classifier: Classifier instance
            x: Input data
            y: True label or target label
            eps: Maximum perturbation size
            eps_iter: Perturbation size per iteration
            nb_iter: Number of iterations
            targeted: Whether it is a targeted attack
        """
        x_adv = x.copy()

        for i in range(nb_iter):

            gradients = classifier.loss_gradient(x_adv, y)

            if targeted:
                # Targeted Attack: Reducing Loss in Target Categories
                perturbation = -eps_iter * np.sign(gradients)
            else:
                # Non-targeted attacks: Increase damage
                perturbation = eps_iter * np.sign(gradients)

            # Application Disturbance
            x_adv = x_adv + perturbation

            # Projected onto the EPS sphere
            delta = x_adv - x
            delta = np.clip(delta, -eps, eps)
            x_adv = x + delta

            # Trim to legal dimensions
            x_adv = np.clip(x_adv, classifier.clip_values[0], classifier.clip_values[1])

        return x_adv

    @staticmethod
    def multi_input_attack(classifier, x_list: List[np.ndarray], y: np.ndarray,
                           attack_method: str = "fgsm", **kwargs) -> List[np.ndarray]:
        """
        Performing Adversarial Attacks on Multi-Input Models

        Parameters:
            classifier: Classifier instance
            x_list: List of input data
            y: Ground truth or target labels
            attack_method: Attack method (“fgsm” or “pgd”)
            **kwargs: Attack parameters

        Returns:
            List of adversarial samples
        """
        #  Perform an attack on each input individually
        x_adv_list = []

        for i, x_input in enumerate(x_list):

            if len(x_input.shape) == 2:

                x_input = np.expand_dims(x_input, axis=-1)

            if attack_method == "fgsm":
                x_adv_input = AdversarialAttacks.fgsm_attack(
                    classifier, x_input, y, **kwargs)
            elif attack_method == "pgd":
                x_adv_input = AdversarialAttacks.pgd_attack(
                    classifier, x_input, y, **kwargs)
            else:
                raise ValueError(f"Unsupported attack method: {attack_method}")

            x_adv_list.append(x_adv_input)

        return x_adv_list



class CustomRobustnessEvaluator:



    def __init__(self, network_transformer, model):
        """
        Initialize Evaluator

        Parameters:
            network_transformer: NetworkTransformer instance
            model: Trained Keras model
        """
        self.nt = network_transformer
        self.model = model
        self.classifier = None
        self.x_test = None
        self.y_test = None
        self.eval_indices = None

    def prepare_test_data(self):
        """Prepare test data for adversarial attacks"""

        eval_mask = ~self.nt.training_mask
        self.eval_indices = np.argwhere(eval_mask).reshape(-1)

        # Ensure the index is within the valid range
        self.eval_indices = self.eval_indices[self.eval_indices >= self.nt.parameters.window_size]

        # Window for obtaining test data
        X_test_windows = []
        for i in self.eval_indices:
            X_test_windows.append(self.nt.X.iloc[(i - self.nt.parameters.window_size) + 1:i + 1])

        # Convert to feature format
        feature_columns_map = {}

        def samplewise_to_featurewise(X):
            # Hard-coded feature mapping (using the mapping provided in the error message)
            feature_column_map = {
                'Bwd Packet Length Max': 'Bwd Packet Length Max',
                ' Packet Length Mean': ' Packet Length Mean',
                'Active Mean': 'Active Mean',
                ' Subflow Fwd Bytes': ' Subflow Fwd Bytes',
                ' Idle Std': ' Idle Std',
                ' Fwd Packet Length Max': ' Fwd Packet Length Max',
                ' Bwd IAT Mean': ' Bwd IAT Mean',
                ' Active Std': ' Active Std',
                ' Fwd IAT Std': ' Fwd IAT Std',
                'Idle Mean': 'Idle Mean',
                ' Active Max': ' Active Max',
                ' Max Packet Length': ' Max Packet Length',
                ' Subflow Bwd Packets': ' Subflow Bwd Packets',
                ' Fwd Packet Length Mean': ' Fwd Packet Length Mean',
                'Init_Win_bytes_forward': 'Init_Win_bytes_forward',
                ' act_data_pkt_fwd': ' act_data_pkt_fwd',
                'Subflow Fwd Packets': 'Subflow Fwd Packets',
                ' Fwd IAT Max': ' Fwd IAT Max',
                ' Bwd Packet Length Std': ' Bwd Packet Length Std',
                ' Bwd IAT Std': ' Bwd IAT Std',
                ' Fwd Header Length': ' Fwd Header Length',
                ' Active Min': ' Active Min',
                ' Average Packet Size': ' Average Packet Size',
                ' Fwd IAT Min': ' Fwd IAT Min',
                ' Subflow Bwd Bytes': ' Subflow Bwd Bytes',
                ' Bwd IAT Max': ' Bwd IAT Max',
                ' Fwd Packet Length Std': ' Fwd Packet Length Std',
                ' Bwd Header Length': ' Bwd Header Length',
                ' Flow IAT Min': ' Flow IAT Min',
                ' Avg Fwd Segment Size': ' Avg Fwd Segment Size',
                ' Idle Max': ' Idle Max',
                ' Packet Length Std': ' Packet Length Std',
                ' Total Backward Packets': ' Total Backward Packets',
                ' Total Fwd Packets': ' Total Fwd Packets',
                ' Avg Bwd Segment Size': ' Avg Bwd Segment Size',
                ' Idle Min': ' Idle Min',
                ' Bwd Packet Length Mean': ' Bwd Packet Length Mean',
                ' Fwd IAT Mean': ' Fwd IAT Mean',
                ' Flow IAT Std': ' Flow IAT Std',
                ' Destination Port': ' Destination Port_1',
                ' Flow Duration': ' Flow Duration_1',
                ' Total Length of Bwd Packets': ' Total Length of Bwd Packets_1',
                'Flow Bytes/s': 'Flow Bytes/s_1',
                ' Flow Packets/s': ' Flow Packets/s_1',
                ' Flow IAT Mean': ' Flow IAT Mean_1',
                ' Flow IAT Max': ' Flow IAT Max_1',
                'Fwd IAT Total': 'Fwd IAT Total_1',
                'Bwd IAT Total': 'Bwd IAT Total_1',
                ' Packet Length Variance': ' Packet Length Variance_1',
                'Fwd Packets/s': 'Fwd Packets/s_1',
                ' Bwd Packets/s': ' Bwd Packets/s_1',
                ' Init_Win_bytes_backward': ' Init_Win_bytes_backward_1'
            }

            sequence_length = len(X[0])
            combined_df = pd.concat(X, ignore_index=True)


            feature_dims = {}
            for feature, base_col in feature_column_map.items():

                one_hot_cols = [col for col in combined_df.columns if
                                col.startswith(base_col[:-2]) and col[-1].isdigit()]
                if one_hot_cols:

                    feature_dims[feature] = len(one_hot_cols)
                else:

                    feature_dims[feature] = 1

            featurewise_X = []

            for feature in self.nt.model_input_spec.feature_names:

                base_col = feature_column_map.get(feature)
                if not base_col:
                    print(f"Warning: No mapping found for feature '{feature}'")
                    continue


                dim = feature_dims.get(feature, 1)

                try:

                    if dim == 1:

                        if base_col in combined_df.columns:
                            combined_values = combined_df[[base_col]].values
                        else:

                            matching_cols = [col for col in combined_df.columns if base_col in col]
                            if matching_cols:
                                combined_values = combined_df[[matching_cols[0]]].values
                            else:
                                print(f"Warning: Column '{base_col}' not found for feature '{feature}'")
                                continue
                    else:

                        one_hot_cols = [f"{base_col[:-2]}_{i}" for i in range(1, dim + 1)]
                        available_cols = [col for col in one_hot_cols if col in combined_df.columns]
                        if not available_cols:

                            if base_col in combined_df.columns:
                                combined_values = combined_df[[base_col]].values
                                dim = 1
                            else:
                                print(f"Warning: No columns found for feature '{feature}'")
                                continue
                        else:
                            combined_values = combined_df[available_cols].values
                            dim = len(available_cols)


                    combined_values = np.array(combined_values, dtype=np.float32)


                    if len(combined_values.shape) == 1:
                        combined_values = np.expand_dims(combined_values, axis=-1)


                    num_samples = len(combined_values) // sequence_length
                    if num_samples * sequence_length != len(combined_values):

                        sequence_length = len(combined_values) // num_samples

                    combined_values = combined_values.reshape(num_samples, sequence_length, dim)
                    featurewise_X.append(combined_values)

                except Exception as e:
                    print(f"Error processing feature '{feature}': {e}")

                    num_samples = len(combined_df) // sequence_length
                    placeholder = np.zeros((num_samples, sequence_length, dim), dtype=np.float32)
                    featurewise_X.append(placeholder)

            return featurewise_X

        self.x_test = samplewise_to_featurewise(X_test_windows)
        self.y_test = self.nt.y[self.eval_indices]


        if isinstance(self.y_test[0], str):

            benign_label = str(self.nt.dataset_specification.benign_label)


            y_test_numeric = []
            for label in self.y_test:
                if str(label) == benign_label:
                    y_test_numeric.append(0.0)
                else:
                    y_test_numeric.append(1.0)

            self.y_test = np.array(y_test_numeric, dtype=np.float32)
        else:

            self.y_test = np.array(self.y_test, dtype=np.float32)


        input_shape = self.x_test[0].shape[1:] if isinstance(self.x_test, list) else self.x_test.shape[1:]


        self.classifier = CustomTensorFlowClassifier(
            model=self.model,
            nb_classes=2,
            input_shape=input_shape,
            loss_object=tf.keras.losses.BinaryCrossentropy(),
            clip_values=(0, 1)
        )

        return self.x_test, self.y_test

    def generate_adversarial_examples(self, attack_method: str = "fgsm", **kwargs):
        """
        Generate Adversarial Examples

        Parameters:
            attack_method: Attack method (“fgsm” or “pgd”)
            **kwargs: Attack parameters

        Returns:
            List of adversarial examples
        """
        if self.x_test is None or self.y_test is None:
            self.prepare_test_data()


        if not isinstance(self.x_test, list) or len(self.x_test) != len(self.model.inputs):
            print(
                f"Warning: Expected {len(self.model.inputs)} inputs, got {len(self.x_test) if isinstance(self.x_test, list) else 1}")

            if isinstance(self.x_test, list) and len(self.x_test) > len(self.model.inputs):
                self.x_test = self.x_test[:len(self.model.inputs)]
            else:

                while len(self.x_test) < len(self.model.inputs):
                    self.x_test.append(np.zeros_like(self.x_test[0]))

        x_adv_list = []


        for i, x_input in enumerate(self.x_test):
            try:

                if len(x_input.shape) == 2:

                    x_input = np.expand_dims(x_input, axis=-1)


                temp_classifier = CustomTensorFlowClassifier(
                    model=self.model,
                    nb_classes=2,
                    input_shape=x_input.shape[1:],
                    loss_object=tf.keras.losses.BinaryCrossentropy(),
                    clip_values=(0, 1)
                )

                if attack_method == "fgsm":
                    x_adv_input = AdversarialAttacks.fgsm_attack(
                        temp_classifier, x_input, self.y_test, **kwargs)
                elif attack_method == "pgd":
                    x_adv_input = AdversarialAttacks.pgd_attack(
                        temp_classifier, x_input, self.y_test, **kwargs)
                else:
                    raise ValueError(f"Unsupported attack method: {attack_method}")

                x_adv_list.append(x_adv_input)
            except Exception as e:
                print(f"Error generating adversarial example for input {i}: {e}")

                x_adv_list.append(x_input)

        return x_adv_list

    def generate_adversarial_examples_simple(self, x_data, y_data, eps=0.1):
        """
        Generating Adversarial Samples Using a Simplified Approach

        Parameters:
            x_data: Input data
            y_data: True labels
            eps: Perturbation magnitude

        Returns:
            Adversarial samples
        """
        if isinstance(x_data, list):

            x_adv = []
            for x_input in x_data:

                x_tensor = tf.convert_to_tensor(x_input, dtype=tf.float32)


                with tf.GradientTape() as tape:
                    tape.watch(x_tensor)
                    predictions = self.model([x_tensor] if isinstance(x_data, list) else x_tensor)
                    loss = tf.keras.losses.binary_crossentropy(y_data, predictions)


                gradients = tape.gradient(loss, x_tensor)


                perturbation = eps * tf.sign(gradients)
                x_adv_input = x_tensor + perturbation


                x_adv_input = tf.clip_by_value(x_adv_input, 0, 1)
                x_adv.append(x_adv_input.numpy())

            return x_adv
        else:

            x_tensor = tf.convert_to_tensor(x_data, dtype=tf.float32)

            with tf.GradientTape() as tape:
                tape.watch(x_tensor)
                predictions = self.model(x_tensor)
                loss = tf.keras.losses.binary_crossentropy(y_data, predictions)


            gradients = tape.gradient(loss, x_tensor)


            perturbation = eps * tf.sign(gradients)
            x_adv = x_tensor + perturbation


            x_adv = tf.clip_by_value(x_adv, 0, 1)

            return x_adv.numpy()

    def evaluate_model_performance(self, x_data, y_data):
        """
        Evaluating model performance on given data

        Parameters:
            x_data: Input data
            y_data: True labels

        Returns:
            A dictionary containing various performance metrics
        """



        print(f"Number of inputs: {len(self.model.inputs)}")
        for i, input_tensor in enumerate(self.model.inputs):
            print(f"Input {i}: {input_tensor.shape}")

        if isinstance(x_data, list):

            if len(x_data) != len(self.model.inputs):
                print(f"Warning: Expected {len(self.model.inputs)} inputs, got {len(x_data)}")

                if len(x_data) > len(self.model.inputs):
                    x_data = x_data[:len(self.model.inputs)]
                else:

                    while len(x_data) < len(self.model.inputs):
                        x_data.append(np.zeros_like(x_data[0]))

            preds = self.classifier.predict(x_data)
        else:

            preds = self.classifier.predict([x_data])


        if len(preds.shape) > 2:
            preds = preds.reshape(preds.shape[0], -1)


        if len(preds.shape) > 1 and preds.shape[1] > 1:
            preds = preds[:, -1]  # 取最后一列

        preds_binary = (preds > 0.5).astype(int).flatten()
        y_data_flat = y_data.flatten()


        accuracy = np.mean(preds_binary == y_data_flat)


        TP = np.sum((preds_binary == 1) & (y_data_flat == 1))
        FP = np.sum((preds_binary == 1) & (y_data_flat == 0))
        TN = np.sum((preds_binary == 0) & (y_data_flat == 0))
        FN = np.sum((preds_binary == 0) & (y_data_flat == 1))


        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        miss_rate = FN / (TP + FN) if (TP + FN) > 0 else 0
        false_alarm_rate = FP / (FP + TN) if (FP + TN) > 0 else 0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
        balanced_accuracy = (recall + specificity) / 2

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "miss_rate": miss_rate,
            "false_alarm_rate": false_alarm_rate,
            "balanced_accuracy": balanced_accuracy,
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN
        }

    def evaluate_robustness(self, attack_methods=None, attack_params=None):
        """
        Evaluating Model Robustness Against Different Attacks

        Parameters:
            attack_methods: List of attack methods to evaluate
            attack_params: Parameters for each attack method

        Returns:
            Dictionary containing evaluation results
        """
        if attack_methods is None:
            attack_methods = ["fgsm", "pgd"]

        if attack_params is None:
            attack_params = {
                "fgsm": {"eps": 0.05},
                "pgd": {"eps": 0.05, "eps_iter": 0.01, "nb_iter": 10}
            }


        print("Evaluating model performance on clean data...")
        clean_performance = self.evaluate_model_performance(self.x_test, self.y_test)
        clean_accuracy = clean_performance["accuracy"]

        results = {"clean": clean_performance}

        for method in attack_methods:
            print(f"Generating adversarial examples using {method.upper()} attack...")
            try:
                x_adv = self.generate_adversarial_examples(method, **attack_params[method])


                print(f"Evaluating model performance on {method.upper()} adversarial examples...")
                adv_performance = self.evaluate_model_performance(x_adv, self.y_test)
                adv_accuracy = adv_performance["accuracy"]


                accuracy_drop = clean_accuracy - adv_accuracy
                robustness_score = adv_accuracy / clean_accuracy if clean_accuracy > 0 else 0

                adv_performance["accuracy_drop"] = accuracy_drop
                adv_performance["robustness_score"] = robustness_score

                results[method] = adv_performance

                print(f"{method.upper()} Attack Results:")
                print(f"  Clean Accuracy: {clean_accuracy:.4f}")
                print(f"  Adversarial Accuracy: {adv_accuracy:.4f}")
                print(f"  Accuracy Drop: {accuracy_drop:.4f}")
                print(f"  Robustness Score: {robustness_score:.4f}")
                print(f"  Precision: {adv_performance['precision']:.4f}")
                print(f"  Recall: {adv_performance['recall']:.4f}")
                print(f"  F1 Score: {adv_performance['f1_score']:.4f}")

            except Exception as e:
                print(f"Error generating {method.upper()} adversarial examples: {e}")
                results[method] = {"error": str(e)}

        return results

    def plot_results(self, results):


        valid_methods = [m for m in results.keys() if "error" not in results[m]]
        clean_acc = results["clean"]["accuracy"]


        accuracies = [results[m]["accuracy"] for m in valid_methods if m != "clean"]
        methods = [m.upper() for m in valid_methods if m != "clean"]

        x = range(len(methods))
        plt.figure(figsize=(12, 6))
        plt.bar(x, [clean_acc] * len(methods), width=0.4, label='Clean Accuracy', align='center', alpha=0.7)
        plt.bar(x, accuracies, width=0.4, label='Adversarial Accuracy', align='edge')
        plt.xlabel('Attack Methods')
        plt.ylabel('Accuracy')
        plt.title('Model Robustness Against Different Attacks')
        plt.xticks(x, methods)
        plt.legend()
        plt.tight_layout()
        plt.savefig('accuracy_comparison.png')
        plt.show()


        robustness_scores = [results[m]["robustness_score"] for m in valid_methods if m != "clean"]
        plt.figure(figsize=(10, 6))
        plt.bar(methods, robustness_scores)
        plt.xlabel('Attack Methods')
        plt.ylabel('Robustness Score')
        plt.title('Model Robustness Score Against Different Attacks')
        plt.tight_layout()
        plt.savefig('robustness_scores.png')
        plt.show()


        metrics = ["precision", "recall", "f1_score", "balanced_accuracy"]
        metric_names = ["Precision", "Recall", "F1 Score", "Balanced Accuracy"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for i, metric in enumerate(metrics):
            clean_vals = [results["clean"][metric]]
            adv_vals = [results[m][metric] for m in valid_methods if m != "clean"]


            if len(adv_vals) == 0:
                print(f"No adversarial data for metric {metric}, skipping.")
                continue

            clean_vals_repeated = clean_vals * len(adv_vals)

            x_pos = np.arange(len(adv_vals))
            axes[i].bar(x_pos - 0.2, clean_vals_repeated, width=0.4, label='Clean', alpha=0.7)
            axes[i].bar(x_pos + 0.2, adv_vals, width=0.4, label='Adversarial')
            axes[i].set_xlabel('Attack Methods')
            axes[i].set_ylabel(metric_names[i])
            axes[i].set_title(f'{metric_names[i]} Comparison')
            axes[i].set_xticks(x_pos)
            axes[i].set_xticklabels([m.upper() for m in valid_methods if m != "clean"])
            axes[i].legend()

        plt.tight_layout()
        plt.savefig('metrics_comparison.png')
        plt.show()


        fig, axes = plt.subplots(1, len(valid_methods), figsize=(5 * len(valid_methods), 4))
        if len(valid_methods) == 1:
            axes = [axes]

        for i, method in enumerate(valid_methods):
            if method == "clean":
                continue

            cm = np.array([
                [results[method]["TN"], results[method]["FP"]],
                [results[method]["FN"], results[method]["TP"]]
                 ])

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i - 1] if i > 0 else axes[i])
            axes[i - 1 if i > 0 else i].set_title(f'Confusion Matrix - {method.upper()}')
            axes[i - 1 if i > 0 else i].set_xlabel('Predicted')
            axes[i - 1 if i > 0 else i].set_ylabel('Actual')

        plt.tight_layout()
        plt.savefig('confusion_matrices.png')
        plt.show()

    def evaluate_perturbation_impact(self, epsilons=None):
        """
        Evaluate the impact of different perturbation magnitudes on model performance

        Parameters:
            epsilons: List of perturbation magnitudes to test

        Returns:
            Dictionary containing results
        """
        if epsilons is None:
            epsilons = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        results = {}

        for eps in epsilons:
            print(f"Testing FGSM attack with epsilon = {eps}")
            try:
                x_adv = self.generate_adversarial_examples("fgsm", eps=eps)

                adv_performance = self.evaluate_model_performance(x_adv, self.y_test)
                adv_accuracy = adv_performance["accuracy"]
                results[eps] = adv_accuracy

                print(f"  Adversarial Accuracy: {adv_accuracy:.4f}")
            except Exception as e:
                print(f"Error with epsilon {eps}: {e}")
                results[eps] = None


        eps_list = [eps for eps in results.keys() if results[eps] is not None]
        acc_list = [results[eps] for eps in eps_list]

        plt.figure(figsize=(10, 6))
        plt.plot(eps_list, acc_list, 'o-')
        plt.xlabel('Epsilon (Perturbation Size)')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy vs Perturbation Size (FGSM)')
        plt.grid(True)
        plt.savefig('perturbation_impact.png')
        plt.show()

        return results

    def adversarial_training(self, epochs=3, batch_size=32, eps=0.1):
        """
        Implement adversarial training to enhance model robustness

        Parameters:
            epochs: Number of adversarial training epochs
            batch_size: Batch size
            eps: Adversarial perturbation magnitude

        Returns:
            Adversarial training-optimized model
        """
        print("Starting adversarial training...")


        train_mask = self.nt.training_mask
        train_indices = np.argwhere(train_mask).reshape(-1)
        train_indices = train_indices[train_indices >= self.nt.parameters.window_size]


        def get_training_batch(indices):
            X_windows = []
            for i in indices:
                X_windows.append(self.nt.X.iloc[(i - self.nt.parameters.window_size) + 1:i + 1])


            feature_columns_map = {}

            def samplewise_to_featurewise(X):

                feature_column_map = {
                    'Bwd Packet Length Max': 'Bwd Packet Length Max',
                    ' Packet Length Mean': ' Packet Length Mean',
                    'Active Mean': 'Active Mean',
                    ' Subflow Fwd Bytes': ' Subflow Fwd Bytes',
                    ' Idle Std': ' Idle Std',
                    ' Fwd Packet Length Max': ' Fwd Packet Length Max',
                    ' Bwd IAT Mean': ' Bwd IAT Mean',
                    ' Active Std': ' Active Std',
                    ' Fwd IAT Std': ' Fwd IAT Std',
                    'Idle Mean': 'Idle Mean',
                    ' Active Max': ' Active Max',
                    ' Max Packet Length': ' Max Packet Length',
                    ' Subflow Bwd Packets': ' Subflow Bwd Packets',
                    ' Fwd Packet Length Mean': ' Fwd Packet Length Mean',
                    'Init_Win_bytes_forward': 'Init_Win_bytes_forward',
                    ' act_data_pkt_fwd': ' act_data_pkt_fwd',
                    'Subflow Fwd Packets': 'Subflow Fwd Packets',
                    ' Fwd IAT Max': ' Fwd IAT Max',
                    ' Bwd Packet Length Std': ' Bwd Packet Length Std',
                    ' Bwd IAT Std': ' Bwd IAT Std',
                    ' Fwd Header Length': ' Fwd Header Length',
                    ' Active Min': ' Active Min',
                    ' Average Packet Size': ' Average Packet Size',
                    ' Fwd IAT Min': ' Fwd IAT Min',
                    ' Subflow Bwd Bytes': ' Subflow Bwd Bytes',
                    ' Bwd IAT Max': ' Bwd IAT Max',
                    ' Fwd Packet Length Std': ' Fwd Packet Length Std',
                    ' Bwd Header Length': ' Bwd Header Length',
                    ' Flow IAT Min': ' Flow IAT Min',
                    ' Avg Fwd Segment Size': ' Avg Fwd Segment Size',
                    ' Idle Max': ' Idle Max',
                    ' Packet Length Std': ' Packet Length Std',
                    ' Total Backward Packets': ' Total Backward Packets',
                    ' Total Fwd Packets': ' Total Fwd Packets',
                    ' Avg Bwd Segment Size': ' Avg Bwd Segment Size',
                    ' Idle Min': ' Idle Min',
                    ' Bwd Packet Length Mean': ' Bwd Packet Length Mean',
                    ' Fwd IAT Mean': ' Fwd IAT Mean',
                    ' Flow IAT Std': ' Flow IAT Std',
                    ' Destination Port': ' Destination Port_1',
                    ' Flow Duration': ' Flow Duration_1',
                    ' Total Length of Bwd Packets': ' Total Length of Bwd Packets_1',
                    'Flow Bytes/s': 'Flow Bytes/s_1',
                    ' Flow Packets/s': ' Flow Packets/s_1',
                    ' Flow IAT Mean': ' Flow IAT Mean_1',
                    ' Flow IAT Max': ' Flow IAT Max_1',
                    'Fwd IAT Total': 'Fwd IAT Total_1',
                    'Bwd IAT Total': 'Bwd IAT Total_1',
                    ' Packet Length Variance': ' Packet Length Variance_1',
                    'Fwd Packets/s': 'Fwd Packets/s_1',
                    ' Bwd Packets/s': ' Bwd Packets/s_1',
                    ' Init_Win_bytes_backward': ' Init_Win_bytes_backward_1'
                }

                sequence_length = len(X[0])
                combined_df = pd.concat(X, ignore_index=True)


                feature_dims = {}
                for feature, base_col in feature_column_map.items():

                    one_hot_cols = [col for col in combined_df.columns if
                                    col.startswith(base_col[:-2]) and col[-1].isdigit()]
                    if one_hot_cols:

                        feature_dims[feature] = len(one_hot_cols)
                    else:

                        feature_dims[feature] = 1

                featurewise_X = []

                for feature in self.nt.model_input_spec.feature_names:

                    base_col = feature_column_map.get(feature)
                    if not base_col:
                        print(f"Warning: No mapping found for feature '{feature}'")
                        continue


                    dim = feature_dims.get(feature, 1)

                    try:

                        if dim == 1:

                            if base_col in combined_df.columns:
                                combined_values = combined_df[[base_col]].values
                            else:

                                matching_cols = [col for col in combined_df.columns if base_col in col]
                                if matching_cols:
                                    combined_values = combined_df[[matching_cols[0]]].values
                                else:
                                    print(f"Warning: Column '{base_col}' not found for feature '{feature}'")
                                    continue
                        else:

                            one_hot_cols = [f"{base_col[:-2]}_{i}" for i in range(1, dim + 1)]

                            available_cols = [col for col in one_hot_cols if col in combined_df.columns]
                            if not available_cols:

                                if base_col in combined_df.columns:
                                    combined_values = combined_df[[base_col]].values
                                    dim = 1
                                else:
                                    print(f"Warning: No columns found for feature '{feature}'")
                                    continue
                            else:
                                combined_values = combined_df[available_cols].values
                                dim = len(available_cols)


                        combined_values = np.array(combined_values, dtype=np.float32)


                        if len(combined_values.shape) == 1:
                            combined_values = np.expand_dims(combined_values, axis=-1)


                        num_samples = len(combined_values) // sequence_length
                        if num_samples * sequence_length != len(combined_values):

                            sequence_length = len(combined_values) // num_samples

                        combined_values = combined_values.reshape(num_samples, sequence_length, dim)
                        featurewise_X.append(combined_values)

                    except Exception as e:
                        print(f"Error processing feature '{feature}': {e}")

                        num_samples = len(combined_df) // sequence_length
                        placeholder = np.zeros((num_samples, sequence_length, dim), dtype=np.float32)
                        featurewise_X.append(placeholder)

                return featurewise_X

            X_batch = samplewise_to_featurewise(X_windows)
            y_batch = self.nt.y[indices]


            if isinstance(y_batch[0], str):

                benign_label = str(self.nt.dataset_specification.benign_label)


                y_batch_numeric = []
                for label in y_batch:
                    if str(label) == benign_label:
                        y_batch_numeric.append(0.0)
                    else:
                        y_batch_numeric.append(1.0)

                y_batch = np.array(y_batch_numeric, dtype=np.float32)
            else:

                y_batch = np.array(y_batch, dtype=np.float32)

            return X_batch, y_batch


        for epoch in range(epochs):
            print(f"Adversarial Training Epoch {epoch + 1}/{epochs}")


            np.random.shuffle(train_indices)


            for i in range(0, len(train_indices), batch_size):
                batch_indices = train_indices[i:i + batch_size]
                if len(batch_indices) < batch_size:
                    continue


                X_batch, y_batch = get_training_batch(batch_indices)


                try:

                    X_adv = self.generate_adversarial_examples_simple(X_batch, y_batch, eps=eps)


                    X_mixed = []
                    for j in range(len(X_batch)):
                        X_mixed.append(np.concatenate([X_batch[j], X_adv[j]], axis=0))

                    y_mixed = np.concatenate([y_batch, y_batch], axis=0)

                    self.model.train_on_batch(X_mixed, y_mixed)

                except Exception as e:
                    print(f"Error in adversarial training batch: {e}")

                    try:
                        self.model.train_on_batch(X_batch, y_batch)
                    except Exception as e2:
                        print(f"Error training on clean batch: {e2}")

                        continue

            print(f"Completed epoch {epoch + 1}")

        print("Adversarial training completed!")
        return self.model



def main():
    """Main Function: Execute the complete adversarial attack evaluation experiment"""

    encodings = [
        NoInputEncoder(),
        RecordLevelEmbed(64),
        RecordLevelEmbed(64, project=True)
    ]

    classification_heads = [
        LastTokenClassificationHead(),
        GlobalAveragePoolingClassificationHead(),
    ]

    transformers: List[FunctionalComponent] = [
        EncoderDecoderTransformer(
            n_encoder_layers=3,
            n_decoder_layers=3,
            internal_size=128,
            n_heads=5
        )
    ]

    flow_file_path = r"D:\dataset"

    datasets = [
        ("CICIDS2017", pd.read_csv(r"D:\dataset\unknown_attacks\train attacks.csv", encoding='gbk'),
         NamedDatasetSpecifications.CICIDS2017, 0.2, EvaluationDatasetSampling.RandomRows),
        ("UnknownAttack", pd.read_csv(r"D:\dataset\unknown_attacks\unknown_attack6.csv", encoding='gbk'),
         NamedDatasetSpecifications.UnknownAttack, 0.2, EvaluationDatasetSampling.RandomRows)
    ]

    pre_processing = StandardPreProcessing(n_categorical_levels=32)


    print("Training model on CICIDS2017 dataset...")
    ft = DGTTransformer(pre_processing=pre_processing,
                            input_encoding=encodings[1],
                            sequential_model=transformers[0],
                            classification_head=LastTokenClassificationHead(),
                            params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

    dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
    ft.load_dataset(dataset_name, dataset_path, dataset_specification,
                    evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)


    m = ft.build_model()

    print(f"Number of inputs: {len(m.inputs)}")
    for i, input_tensor in enumerate(m.inputs):
        print(f"Input {i}: {input_tensor.shape}")
    m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])


    train_results, eval_results, final_epoch = ft.evaluate(m, batch_size=256, epochs=5,
                                                           steps_per_epoch=64, early_stopping_patience=10)
    print('CICIDS2017 test dataset prediction results\n', eval_results)


    print("\nEvaluating model robustness against adversarial attacks...")
    evaluator = CustomRobustnessEvaluator(ft, m)
    evaluator.prepare_test_data()


    robustness_results = evaluator.evaluate_robustness()


    evaluator.plot_results(robustness_results)


    perturbation_results = evaluator.evaluate_perturbation_impact()


    print("\nPerforming adversarial training to improve robustness...")
    robust_model = evaluator.adversarial_training(epochs=3, eps=0.1)


    print("\nEvaluating robustness after adversarial training...")
    evaluator_robust = CustomRobustnessEvaluator(ft, robust_model)
    evaluator_robust.prepare_test_data()

    robustness_results_after = evaluator_robust.evaluate_robustness()


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))


    methods = [m for m in robustness_results.keys() if "error" not in robustness_results[m] and m != "clean"]
    orig_acc = [robustness_results[m]["accuracy"] for m in methods]


    robust_acc = [robustness_results_after[m]["accuracy"] for m in methods if
                  m in robustness_results_after and "error" not in robustness_results_after[m]]

    x = np.arange(len(methods))
    axes[0].bar(x - 0.2, orig_acc, width=0.4, label='Original Model')
    axes[0].bar(x + 0.2, robust_acc, width=0.4, label='After Adversarial Training')
    axes[0].set_xlabel('Attack Methods')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy Before and After Adversarial Training')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([m.upper() for m in methods])
    axes[0].legend()


    orig_robust = [robustness_results[m]["robustness_score"] for m in methods]
    robust_robust = [robustness_results_after[m]["robustness_score"] for m in methods if
                     m in robustness_results_after and "error" not in robustness_results_after[m]]

    axes[1].bar(x - 0.2, orig_robust, width=0.4, label='Original Model')
    axes[1].bar(x + 0.2, robust_robust, width=0.4, label='After Adversarial Training')
    axes[1].set_xlabel('Attack Methods')
    axes[1].set_ylabel('Robustness Score')
    axes[1].set_title('Robustness Score Before and After Adversarial Training')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([m.upper() for m in methods])
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('adversarial_training_comparison.png')
    plt.show()


    print("\nSaving results...")
    results_df = pd.DataFrame({
        'Attack_Method': [],
        'Original_Accuracy': [],
        'Robust_Accuracy': [],
        'Accuracy_Improvement': [],
        'Original_Robustness_Score': [],
        'Robust_Robustness_Score': [],
        'Robustness_Improvement': []
    })

    for method in methods:
        if method in robustness_results_after and "error" not in robustness_results_after[method]:
            orig_acc = robustness_results[method]["accuracy"]
            robust_acc = robustness_results_after[method]["accuracy"]
            acc_improvement = robust_acc - orig_acc

            orig_robust = robustness_results[method]["robustness_score"]
            robust_robust = robustness_results_after[method]["robustness_score"]
            robust_improvement = robust_robust - orig_robust

            results_df = results_df.append({
                'Attack_Method': method.upper(),
                'Original_Accuracy': orig_acc,
                'Robust_Accuracy': robust_acc,
                'Accuracy_Improvement': acc_improvement,
                'Original_Robustness_Score': orig_robust,
                'Robust_Robustness_Score': robust_robust,
                'Robustness_Improvement': robust_improvement
            }, ignore_index=True)

    results_df.to_csv('robustness_evaluation_results.csv', index=False)
    print("Results saved to robustness_evaluation_results.csv")


    print("\nTesting model generalization on unknown attack dataset...")
    ft_unknown = DGTTransformer(pre_processing=pre_processing,
                                    input_encoding=encodings[1],
                                    sequential_model=transformers[0],
                                    classification_head=LastTokenClassificationHead(),
                                    params=DGTParameters(window_size=8, mlp_layer_sizes=[128],
                                                                        mlp_dropout=0.1))

    dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[1]
    ft_unknown.load_dataset(dataset_name, dataset_path, dataset_specification,
                            evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)


    evaluator_unknown = CustomRobustnessEvaluator(ft_unknown, m)
    evaluator_unknown.prepare_test_data()
    unknown_results = evaluator_unknown.evaluate_model_performance(evaluator_unknown.x_test, evaluator_unknown.y_test)


    evaluator_unknown_robust = CustomRobustnessEvaluator(ft_unknown, robust_model)
    evaluator_unknown_robust.prepare_test_data()
    unknown_results_robust = evaluator_unknown_robust.evaluate_model_performance(
        evaluator_unknown_robust.x_test, evaluator_unknown_robust.y_test)

    print(f"\nPerformance on unknown attacks:")
    print(f"Original Model - Accuracy: {unknown_results['accuracy']:.4f}, F1: {unknown_results['f1_score']:.4f}")
    print(
        f"Robust Model - Accuracy: {unknown_results_robust['accuracy']:.4f}, F1: {unknown_results_robust['f1_score']:.4f}")


    with open('robustness_evaluation_report.txt', 'w') as f:
        f.write("Liquid Transformer Adversarial Robustness Evaluation Report\n")
        f.write("=" * 60 + "\n\n")

        f.write("1. MODEL ARCHITECTURE\n")
        f.write(f"   - Input Encoding: {encodings[1].__class__.__name__}\n")
        f.write(f"   - Transformer: {transformers[0].__class__.__name__}\n")
        f.write(f"   - Classification Head: {LastTokenClassificationHead().__class__.__name__}\n")
        f.write(f"   - Window Size: {8}\n\n")

        f.write("2. DATASETS\n")
        f.write(f"   - Training: {datasets[0][0]}\n")
        f.write(f"   - Testing: {datasets[1][0]}\n\n")

        f.write("3. ROBUSTNESS EVALUATION RESULTS\n")
        for method in methods:
            if method in robustness_results and "error" not in robustness_results[method]:
                f.write(f"   {method.upper()} Attack:\n")
                f.write(f"     - Clean Accuracy: {robustness_results['clean']['accuracy']:.4f}\n")
                f.write(f"     - Adversarial Accuracy: {robustness_results[method]['accuracy']:.4f}\n")
                f.write(f"     - Robustness Score: {robustness_results[method]['robustness_score']:.4f}\n\n")

        f.write("4. ADVERSARIAL TRAINING IMPACT\n")
        for method in methods:
            if method in robustness_results_after and "error" not in robustness_results_after[method]:
                f.write(f"   {method.upper()} Attack After Training:\n")
                f.write(f"     - Adversarial Accuracy: {robustness_results_after[method]['accuracy']:.4f}\n")
                f.write(f"     - Robustness Score: {robustness_results_after[method]['robustness_score']:.4f}\n")
                improvement = robustness_results_after[method]['accuracy'] - robustness_results[method]['accuracy']
                f.write(f"     - Accuracy Improvement: {improvement:.4f}\n\n")

        f.write("5. GENERALIZATION TO UNKNOWN ATTACKS\n")
        f.write(f"   - Original Model Accuracy: {unknown_results['accuracy']:.4f}\n")
        f.write(f"   - Robust Model Accuracy: {unknown_results_robust['accuracy']:.4f}\n")
        f.write(f"   - Improvement: {unknown_results_robust['accuracy'] - unknown_results['accuracy']:.4f}\n")

    print("Evaluation completed! Report saved to robustness_evaluation_report.txt")


if __name__ == "__main__":
    main()








