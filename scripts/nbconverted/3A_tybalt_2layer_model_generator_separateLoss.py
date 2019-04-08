
# coding: utf-8

# # Train 2-layer Tybalt 
# By Alexandra Lee
# 
# created December 2018
# 
# Encode Pseudomonas gene expression data into low dimensional latent space using Tybalt with 2-hidden layers.  This time plot loss function separately (i.e. separate reconstruction loss and KL divergence).
# 
# Note: Need to use python 3 to support '*' syntax change

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
from functions.my_classes import DataGenerator
#from functions.vae_utils import VariationalLayer, WarmUpCallback, LossCallback
from functions.models import Tybalt


# In[2]:


# To ensure reproducibility using Keras during development
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import numpy as np
import random as rn

# The below is necessary in Python 3.2.3 onwards to
# have reproducible behavior for certain hash-based operations.
# See these references for further details:
# https://docs.python.org/3.4/using/cmdline.html#envvar-PYTHONHASHSEED
# https://github.com/keras-team/keras/issues/2280#issuecomment-306959926
randomState = 123
import os
os.environ['PYTHONHASHSEED'] = '0'

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# Force TensorFlow to use single thread.
# Multiple threads are a potential source of
# non-reproducible results.
# For further details, see: https://stackoverflow.com/questions/42022950/which-seeds-have-to-be-set-where-to-realize-100-reproducibility-of-training-res

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed

tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

import keras
from keras.layers import Input, Dense, Lambda, Layer, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras import metrics, optimizers
from keras.callbacks import Callback


# ## Initialize hyper parameters
# 
# 1.  learning rate: 
# 2.  batch size: Total number of training examples present in a single batch.  Iterations is the number of batches needed to complete one epoch
# 3.  epochs: One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE
# 4.  kappa: warmup
# 5.  original dim: dimensions of the raw data
# 6.  latent dim: dimensiosn of the latent space (fixed by the user)
#     Note: intrinsic latent space dimension unknown
# 7.  epsilon std: 
# 8.  beta: Threshold value for ReLU?

# In[3]:


# Initialize parameters
learning_rate_model = 0.00001
epochs_model = 100
kappa_model = 0.01

intermediate_dim_model = 100
latent_dim_model = 10
epsilon_std_model = 1.0
beta_model = K.variable(0)

chunk_size = 100

# Get dimensions of datasets
train_dim_file =  os.path.join(
    os.path.dirname(
        os.getcwd()),"metadata", "all_1M_validation_0.1", "train_dim.pickle")

val_dim_file =  os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata", "all_1M_validation_0.1", "validation_dim.pickle")

with open(train_dim_file, 'rb') as f:
    num_train_samples, num_genes = pickle.load(f)
with open(val_dim_file, 'rb') as f:
    num_val_samples, num_genes = pickle.load(f)

original_dim_model = num_genes
print(num_train_samples)#num_samples_train = 1319122
print(num_val_samples)#num_samples_val = 1319


# In[4]:


# Load gene expression data using generator 
train_file =  "/home/alexandra/Documents/Data/LINCS_tuning/train_model_input.txt.xz"
validation_file = "/home/alexandra/Documents/Data/LINCS_tuning/validation_model_input.txt.xz"

training_generator = DataGenerator(train_file, chunk_size, num_train_samples)
validation_generator = DataGenerator(validation_file, chunk_size, num_val_samples)


# In[5]:


# Model 
model = Tybalt(original_dim=original_dim_model,
                 hidden_dim=intermediate_dim_model,
                 latent_dim=latent_dim_model,
                 batch_size=chunk_size,
                 epochs=epochs_model,
                 learning_rate=learning_rate_model,
                 kappa=kappa_model,
                 beta=beta_model,
                 epsilon_std=epsilon_std_model)


# In[6]:


# Output files
stat_file =  os.path.join(
    os.path.dirname(
        os.getcwd()), "stats", "tybalt_2layer_{}latent_stats.tsv".format(latent_dim_model))
hist_plot_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "stats", "tybalt_2layer_{}latent_hist.png".format(latent_dim_model))

encoded_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "encoded", "train_input_2layer_{}latent_encoded.txt".format(latent_dim_model))

model_encoder_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "models", "tybalt_2layer_{}latent_encoder_model.h5".format(latent_dim_model))
weights_encoder_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "models", "tybalt_2layer_{}latent_encoder_weights.h5".format(latent_dim_model))
model_decoder_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "models", "tybalt_2layer_{}latent_decoder_model.h5".format(latent_dim_model))
weights_decoder_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "models", "tybalt_2layer_{}latent_decoder_weights.h5".format(latent_dim_model))


# In[7]:


# Compile Model
model.build_encoder_layer()
model.build_decoder_layer()
model.compile_vae()
model._connect_layers()
model.get_summary()


# In[8]:


# Model architecture
model_architecture_file = os.path.join(
    os.path.dirname(
        os.getcwd()),'stats', 'vae_architecture.png')
model.visualize_architecture(model_architecture_file)


# In[9]:


get_ipython().run_cell_magic('time', '', 'model.train_vae(train_df=training_generator,\n                test_df=validation_generator,\n               separate_loss=True)')


# In[10]:


model_training_file = os.path.join(
    os.path.dirname(
        os.getcwd()), 'stats', 'training.pdf')
model.visualize_training(model_training_file)


# In[11]:


# Plot reconstruction loss
recon_loss = model.history_df['recon']
epochs = range(epochs_model)
plt.figure()
plt.plot(epochs, recon_loss, 'b', label='recon loss')
plt.title('Reconstruction loss')
plt.legend()
plt.show()


# In[12]:


# Plot KL divergence
kl_loss = model.history_df['kl']
epochs = range(epochs_model)
plt.figure()
plt.plot(epochs, kl_loss, 'g', label='KL loss')
plt.title('KL Divergence')
plt.legend()
plt.show()


# In[16]:


# Plot KL divergence
recon_loss = model.history_df['recon']
kl_loss = - model.history_df['kl']
epochs = range(epochs_model)
plt.figure()
plt.plot(epochs, recon_loss, 'b', label='recon loss')
plt.plot(epochs, kl_loss, 'g', label='KL loss')
plt.title('KL Divergence')
plt.legend()
plt.show()


# In[13]:


#model_compression = model.compress(training_generator)
#model_file = os.path.join(os.path.dirname(os.getcwd()), 'data', 'encoded_rnaseq_twohidden_100model.tsv.gz')
#model_compression.to_csv(model_file, sep='\t', compression='gzip')
#model_compression.head(2)


# In[14]:


model_weights = model.get_decoder_weights()


# In[15]:


encoder_model_file = os.path.join(
    os.path.dirname(
        os.getcwd()),'models', 'encoder_twohidden100_vae.hdf5')
decoder_model_file = os.path.join(
    os.path.dirname(
        os.getcwd()),'models', 'decoder_twohidden100_vae.hdf5')
model.save_models(encoder_model_file, decoder_model_file)

