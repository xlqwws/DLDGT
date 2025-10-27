# DLDGT
Dynamic Lifelong Defense: Continual Detection of Unknown Attacks via Adaptation Generative and Dynamic Memory Transformers

Our presents dynamic lifelong defense (DLDGT), a novel framework architected for the continuous detection of unknown attacks. DLDGT integrates two key innovations: an adaptive generation module (DL-VAE) and a dynamic transformer with memory (DGT), which together enable lifelong learning  and defense capabilities. Specifically, DL-VAE is a novel variational autoencoder with dynamic perception capabilities that can adaptively model the non-stationary distribution of network flows. DGT is an innovative detection architecture that employs developed dynamic attention mechanism and continuous memory module. This enables it to continuously absorb knowledge about novel attacks without catastrophic forgetting, thereby maintaining high vigilance against evolving threats. 

Environment Requirements: TensorFlow >=2.1, Keras >=2.6.0, Python=3.9.

First, set up the dataset specification file, dataset_specification.py.

"""
        Define specific dataset formats
        :param include_fields: Fields used as classification criteria
        :param categorical_fields: Fields to be treated as categorical
        :param class_column: Column name containing traffic categories
        :param benign_label: Label e.g., Benign or 0
        :param test_column: Indicates whether the row belongs to the training or test set
"""
<img width="486" height="123" alt="image" src="https://github.com/user-attachments/assets/183b303a-457f-4d00-8c34-dcbd660adb30" />

Second, open and run Generate_adaptive VAE.py to generate unknown flows. Visualizations of the generated data are located in the directory: DLDGT\\figures and generated data.

Third, the generated unknown traffic needs to be adjusted in the data_specification.py file.

Finally, run the main.py file. 

# Define the DGTtransformer
model = DGTTransformer(pre_processing=pre_processing,
                     input_encoding=encodings[1],
                     sequential_model=transformers[0],
                     classification_head=LastTokenClassificationHead(),
                     params=DGTParameters(window_size=8, mlp_layer_sizes=[128], mlp_dropout=0.1))

# Load the specific dataset
dataset_name, dataset_path, dataset_specification, eval_percent, eval_method = datasets[0]
ft.load_dataset(dataset_name, dataset_path, dataset_specification, evaluation_dataset_sampling=eval_method, evaluation_percent=eval_percent)

# Build the transformer model
m = model.build_model()
m.summary()

# Compile the model
m.compile(optimizer="adam", loss='binary_crossentropy', metrics=['binary_accuracy'])

Experiment.py contains all experiments.


