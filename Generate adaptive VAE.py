import keras
import numpy as np
import pandas as pd
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Layer, Reshape
from keras import backend as K
from sklearn.cluster import KMeans
import os
from sklearn.preprocessing import StandardScaler
from keras.losses import logcosh
import time
from tensorflow.keras.layers import RNN
from tensorflow.keras import layers
import tensorflow as tf

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt



class LiquidCell(layers.Layer):
    def __init__(self, units, **kwargs):
        super(LiquidCell, self).__init__(**kwargs)
        self.units = units
        self.state_size = units
        self.output_size = units

    def build(self, input_shape):
        # Obtain Feature Dimensions
        feature_dim = input_shape[-1]

        # Input weights
        self.kernel = self.add_weight(
            shape=(feature_dim, self.units),
            initializer='glorot_uniform',
            name='kernel'
        )
        # Recursive Weight
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='orthogonal',
            name='recurrent_kernel'
        )

        self.bias = self.add_weight(
            shape=(self.units,),
            initializer='zeros',
            name='bias'
        )
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]


        h = K.dot(inputs, self.kernel) + K.dot(prev_output, self.recurrent_kernel) + self.bias
        output = K.tanh(h)

        return output, [output]

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config


class LiquidLayer(layers.RNN):
    def __init__(self, units, **kwargs):
        cell = LiquidCell(units)
        super(LiquidLayer, self).__init__(cell, return_sequences=False, **kwargs)
        self.units = units

    def build(self, input_shape):

        if input_shape[-1] is None:
            raise ValueError("Feature dimension must be defined")
        super(LiquidLayer, self).build(input_shape)


    def call(self, inputs):

        batch_size = tf.shape(inputs)[0]
        return super().call(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
        })
        return config



df = pd.read_csv(r"D:\dataset\CICIDS2017_improved\cicids2017.csv",encoding='gbk')
df = df.dropna(how='any')


features = df.drop(columns=[' Label'])
labels = df[' Label']


print("Inf check:", np.any(np.isinf(features)))
features[np.isinf(features)] = 0


scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features = pd.DataFrame(scaled_features, columns=features.columns)


hiddendim = 79
latentdim = 8


inputs = Input(shape=(features.shape[1],), name='encoder_input')

x = Reshape((1, features.shape[1]))(inputs)
x = LiquidLayer(hiddendim)(x)
x = Dense(hiddendim, activation='relu')(x)
z_mean = Dense(latentdim, name='z_mean')(x)
z_log_var = Dense(latentdim, name='z_log_var')(x)


def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(latentdim,), name='z')([z_mean, z_log_var])


latent_inputs = Input(shape=(latentdim,), name='z_sampling')
x = Dense(hiddendim, activation='relu')(latent_inputs)
x = Reshape((1, hiddendim))(x)
x = LiquidLayer(hiddendim)(x)
outputs = Dense(features.shape[1], activation='linear')(x)

decoder = Model(latent_inputs, outputs, name='decoder')


outputs = decoder(z)
vae = Model(inputs, outputs, name='vae')


reconstruction_loss = logcosh(inputs, outputs) * features.shape[1]
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)


vae.compile(optimizer='adam')
start_time = time.time()
vae.fit(scaled_features, epochs=10, batch_size=128)
print("Training time: %.2f seconds" % (time.time() - start_time))


num_samples = 100000
latent_samples = np.random.normal(loc=0, scale=1, size=(num_samples, latentdim))
generated_features = decoder.predict(latent_samples)


generated_features = scaler.inverse_transform(generated_features)
generated_features = np.round(generated_features)
generated_features = np.clip(generated_features, 0, None)


if np.any(np.isnan(generated_features)):
    print("Warning: Generated data contains NaN values. Replacing with 0.")
    generated_features = np.nan_to_num(generated_features, nan=0)


most_common_label = labels.mode().values[0]
synthetic_labels = pd.Series([most_common_label] * num_samples)


synthetic_data = pd.DataFrame(
    generated_features,
    columns=features.columns
)
synthetic_data[' Label'] = synthetic_labels.values



synthetic_data.to_csv('D:\dataset\CICIDS2017_improved\generated 10,000-Other attacks.csv', index=False)
print("Generated data saved to generated attack_data.csv.csv")
print("generated attack data preview:")
print(synthetic_data.head())


def clean_data(data):


    data = np.where(np.isinf(data), 0, data)


    data = np.where(np.isnan(data), 0, data)


    max_val = np.finfo(np.float64).max
    min_val = np.finfo(np.float64).min
    data = np.clip(data, min_val, max_val)

    return data




def evaluate_generated_attacks(real_data, synthetic_data):

    if np.any(np.isinf(real_data)) or np.any(np.isnan(real_data)):

        real_data = np.nan_to_num(real_data, nan=0, posinf=0, neginf=0)

    if np.any(np.isinf(synthetic_data)) or np.any(np.isnan(synthetic_data)):

        synthetic_data = np.nan_to_num(synthetic_data, nan=0, posinf=0, neginf=0)


    max_val = np.finfo(np.float64).max
    min_val = np.finfo(np.float64).min

    if np.any(real_data > max_val) or np.any(real_data < min_val):

        real_data = np.clip(real_data, min_val, max_val)

    if np.any(synthetic_data > max_val) or np.any(synthetic_data < min_val):

        synthetic_data = np.clip(synthetic_data, min_val, max_val)


    def clean_array(arr):

        arr = np.nan_to_num(arr, nan=0, posinf=0, neginf=0)


        max_val = np.finfo(np.float64).max
        min_val = np.finfo(np.float64).min
        return np.clip(arr, min_val, max_val)


    real_data = clean_data(real_data)
    synthetic_data = clean_data(synthetic_data)


    n_real = real_data.shape[0]
    n_syn = synthetic_data.shape[0]

    if n_real < 30 or n_syn < 30:
        print(f"Insufficient samples: Real samples = {n_real}, Generated samples = {n_syn}")
        return {
            'diversity_score': -1,
            'feature_correlation': -1,
            'distribution_plots': None
        }


    perplexity_real = min(30, n_real - 1)
    perplexity_syn = min(30, n_syn - 1)

    print(f"Confusion level used: real data={perplexity_real}, generated data={perplexity_syn}")


    try:
        tsne_real = TSNE(n_components=2, perplexity=perplexity_real, random_state=42)
        tsne_syn = TSNE(n_components=2, perplexity=perplexity_syn, random_state=42)

        real_emb = tsne_real.fit_transform(real_data)
        syn_emb = tsne_syn.fit_transform(synthetic_data)
    except Exception as e:
        print(f"t-SNE error: {e}")

        from sklearn.decomposition import PCA
        print("Using PCA as an alternative to t-SNE")
        pca = PCA(n_components=2)
        real_emb = pca.fit_transform(real_data)
        syn_emb = pca.fit_transform(synthetic_data)


    plt.figure(figsize=(12, 6))


    plt.subplot(121)
    plt.scatter(real_emb[:, 0], real_emb[:, 1], alpha=0.5, label='Real Attacks')
    plt.title('Real Attack Distribution-Other attacks')


    plt.subplot(122)
    plt.scatter(syn_emb[:, 0], syn_emb[:, 1], alpha=0.5, color='#B913FF', label='Synthetic Attacks')
    plt.title('Generated Unknown Attack Distribution-Other attacks')


    output_dir = 'D://model code//design paper of conference//figures and generated data//'
    os.makedirs(output_dir, exist_ok=True)


    plot_path = os.path.join(output_dir, 'Attack Distribution  10,000-Other attacks.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Comparative distribution maps have been saved to: {plot_path}")



    stat_corr = -1
    if real_data.shape[1] == synthetic_data.shape[1]:
        try:

            real_means = np.mean(real_data, axis=0)
            syn_means = np.mean(synthetic_data, axis=0)
            stat_corr = np.corrcoef(real_means, syn_means)[0, 1]
        except Exception as e:
            print(f"Feature correlation calculation error: {e}")

    return {

        'feature_correlation': stat_corr,
        'distribution_plots': plot_path
    }


output_dir = 'D://model code//design paper of conference//figures and generated data//'
os.makedirs(output_dir, exist_ok=True)



attack_type = 'Other attacks'
print(f"Type of attack being evaluated: {attack_type}")


real_attacks = df[df[' Label'] == attack_type].drop(' Label', axis=1).values
print(f"Real'{attack_type}' attack sample size: {real_attacks.shape[0]}")


syn_attacks = synthetic_data[synthetic_data[' Label'] == attack_type].drop(' Label', axis=1).values
print(f"Generated '{attack_type}' attack sample size: {syn_attacks.shape[0]}")


if real_attacks.shape[0] >= 30 and syn_attacks.shape[0] >= 30:

    sample_size = min(5000, real_attacks.shape[0], syn_attacks.shape[0])
    results = evaluate_generated_attacks(
        real_attacks[:sample_size],
        syn_attacks[:sample_size]
    )
    print("Evaluation Results:")
    print(f"Diversity Score: {results['diversity_score']}")
    print(f"Feature Correlation: {results['feature_correlation']}")
    print(f"Distribution Map Path: {results['distribution_plots']}")
else:
    print(f"Insufficient samples, skip evaluation of  '{attack_type}' attack")
    print(f"Number of real samples: {real_attacks.shape[0]}, Number of generated samples: {syn_attacks.shape[0]}")