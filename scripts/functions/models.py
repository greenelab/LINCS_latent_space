"""
Adopted from
tybalt/models.py
2017 Gregory Way
Functions enabling the construction and usage of Tybalt and ADAGE models
"""

import numpy as np
import pandas as pd

from keras import backend as K
from keras import optimizers
from keras.layers import Input, Dense, Lambda, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import concatenate
from keras.models import Model, Sequential
from keras.regularizers import l1
from keras.utils import plot_model

from functions.vae_utils import CustomVariationalLayer, WarmUpCallback, LossCallback, sampling_maker
from functions.base import VAE, BaseModel


class Tybalt():
    """
    Facilitates the training and output of tybalt model trained on TCGA RNAseq gene expression data
    """

    def __init__(self, original_dim, hidden_dim, latent_dim,
                 batch_size, epochs, learning_rate, kappa, beta, epsilon_std):
        self.original_dim = original_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.kappa = kappa
        self.beta = beta
        self.epsilon_std = epsilon_std

    def build_encoder_layer(self):
        # Input place holder for RNAseq data with specific input size
        self.rnaseq_input = Input(shape=(self.original_dim, ))

        # Input layer is compressed into a mean and log variance vector of size `latent_dim`
        # Each layer is initialized with glorot uniform weights and each step (dense connections, batch norm,
        # and relu activation) are funneled separately
        # Each vector of length `latent_dim` are connected to the rnaseq input tensor
        hidden_dense_linear = Dense(
            self.hidden_dim, kernel_initializer='glorot_uniform')(self.rnaseq_input)
        hidden_dense_batchnorm = BatchNormalization()(hidden_dense_linear)
        hidden_encoded = Activation('relu')(hidden_dense_batchnorm)

        z_mean_dense_linear = Dense(
            self.latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)
        z_mean_dense_batchnorm = BatchNormalization()(z_mean_dense_linear)
        self.z_mean_encoded = Activation('relu')(z_mean_dense_batchnorm)

        z_log_var_dense_linear = Dense(
            self.latent_dim, kernel_initializer='glorot_uniform')(hidden_encoded)
        z_log_var_dense_batchnorm = BatchNormalization()(z_log_var_dense_linear)
        self.z_log_var_encoded = Activation('relu')(z_log_var_dense_batchnorm)

        # return the encoded and randomly sampled z vector
        # Takes two keras layers as input to the custom sampling function layer with a `latent_dim` output
        self.z = Lambda(sampling_maker(self.epsilon_std), output_shape=(self.latent_dim, ))(
            [self.z_mean_encoded, self.z_log_var_encoded])

    def build_decoder_layer(self):
        # The decoding layer is much simpler with a single layer glorot uniform initialized and sigmoid activation
        self.decoder_model = Sequential()
        self.decoder_model.add(
            Dense(self.hidden_dim, activation='relu', input_dim=self.latent_dim))
        self.decoder_model.add(Dense(self.original_dim, activation='sigmoid'))
        self.rnaseq_reconstruct = self.decoder_model(self.z)

    def compile_vae(self):
        adam = optimizers.Adam(lr=self.learning_rate)
        vae_layer = CustomVariationalLayer(self.z_log_var_encoded,
                                           self.z_mean_encoded,
                                           self.original_dim,
                                           self.beta,
                                           'mse')([self.rnaseq_input, self.rnaseq_reconstruct])

        self.vae = Model(self.rnaseq_input, vae_layer)
        self.vae.compile(optimizer=adam, loss=None, loss_weights=[self.beta])

    def _connect_layers(self):
        """
        Make connections between layers to build separate encoder and decoder
        """
        self.encoder = Model(self.rnaseq_input, self.z_mean_encoded)

        decoder_input = Input(shape=(self.latent_dim, ))
        _x_decoded_mean = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded_mean)

    def get_summary(self):
        self.vae.summary()

    def visualize_architecture(self, output_file):
        # Visualize the connections of the custom VAE model
        plot_model(self.vae, to_file=output_file)
        # SVG(model_to_dot(self.vae).create(prog='dot', format='svg'))

    def train_vae(self, train_df, test_df, separate_loss=False):
        """
        Method to train model.
        `separate_loss` instantiates a custom Keras callback that tracks the
        separate contribution of reconstruction and KL divergence loss. Because
        VAEs try to minimize both, it may be informative to track each across
        training separately. The callback processes the training data through
        the current encoder and decoder and therefore requires additional time
        - which is why this is not done by default.
        """
        cbks = [WarmUpCallback(self.beta, self.kappa)]
        if separate_loss:
            print("updating callback loss...")
            # print(type(train_df))
            # print(train_df)
            # print(np.array(next(iter(train_df))))
            # print(np.array(iter(train_df)))
            # print(np.array(train_df.__getitem__))
            # print(train_df.__getitem__)
            tybalt_loss_cbk = LossCallback(training_data=np.array(next(iter(train_df)))[0],
                                           encoder_cbk=self.encoder,
                                           decoder_cbk=self.decoder,
                                           original_dim=self.original_dim)
            cbks += [tybalt_loss_cbk]
            print(tybalt_loss_cbk)
            training_data = np.array(next(iter(train_df)))
            print(training_data)
            print(training_data.shape)
            print(self.encoder)
            print(self.decoder)
            print(self.original_dim)
            print("added callback")
        print("Going to start training...")

        self.hist = self.vae.fit_generator(generator=train_df,
                                           validation_data=test_df,
                                           shuffle=True,
                                           epochs=self.epochs,
                                           callbacks=cbks)
        self.history_df = pd.DataFrame(self.hist.history)

        if separate_loss:
            self.history_df = self.history_df.assign(
                recon=tybalt_loss_cbk.xent_loss)
            self.history_df = self.history_df.assign(
                kl=tybalt_loss_cbk.kl_loss)

    def visualize_training(self, output_file=None):
        # Visualize training performance
        history_df = pd.DataFrame(self.hist.history)
        ax = history_df.plot()
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        fig = ax.get_figure()
        if output_file:
            fig.savefig(output_file)
        else:
            fig.show()

    def compress(self, df):
        # Model to compress input
        self.encoder = Model(self.rnaseq_input, self.z_mean_encoded)

        # Encode rnaseq into the hidden/latent representation - and save output
        encoded_df = self.encoder.predict_on_batch(df)
        encoded_df = pd.DataFrame(encoded_df, columns=range(1, self.latent_dim + 1),
                                  index=rnaseq_df.index)
        return encoded_df

    def get_decoder_weights(self, decoder=True):
        # build a generator that can sample from the learned distribution
        # can generate from any sampled z vector
        decoder_input = Input(shape=(self.latent_dim, ))
        _x_decoded_mean = self.decoder_model(decoder_input)
        self.decoder = Model(decoder_input, _x_decoded_mean)
        weights = []
        if decoder:
            for layer in self.decoder.layers:
                weights.append(layer.get_weights())
        else:
            for layer in self.encoder.layers:
                # Encoder weights must be transposed
                encoder_weights = layer.get_weights()
                encoder_weights = [np.transpose(x) for x in encoder_weights]
                weights.append(encoder_weights)
        return(weights)

    def predict(self, df):
        return self.decoder.predict(np.array(df))

    def save_models(self, encoder_file, decoder_file):
        self.encoder.save(encoder_file)
        self.decoder.save(decoder_file)
