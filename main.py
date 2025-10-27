

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

# Define the DGTtransformer
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

# Get the evaluation results
eval_results: pd.DataFrame
(train_results, eval_results, final_epoch) = ft.evaluate(m, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)


print('CICIDS2017 test dataset prediction results\n',eval_results)

#第二
# Define the DGTtransformer
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

# Get the evaluation results
eval_results1: pd.DataFrame
(train_results1, eval_results1, final_epoch1) = ft1.evaluate(mn, batch_size=256, epochs=5, steps_per_epoch=64, early_stopping_patience=10)

print('unknown attacks test dataset prediction results\n',eval_results1)

