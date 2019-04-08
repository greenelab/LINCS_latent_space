
# coding: utf-8

# # Get dimension of dataset
# 
# Get dimension of feature space for input into 
# This information is used by [3_tybalt_2layer_model_generator.py](3_tybalt_2layer_model_generator.ipynb) to train autoencoder

# In[ ]:


import os
import pandas as pd
import pickle

randomState = 123


# In[2]:


get_ipython().run_cell_magic('time', '', '# Load dataset\nsubsample_dataset = "subsample_13K_validation_0.2"\ntrain_file =  "/home/alexandra/Documents/Data/LINCS_tuning/"+subsample_dataset+"/train_model_input.txt.xz"\nval_file =  "/home/alexandra/Documents/Data/LINCS_tuning/"+subsample_dataset+"/validation_model_input.txt.xz"')


# In[3]:


# Read data
rnaseq_train = pd.read_table(train_file, index_col=0, header=0, compression = 'xz')
rnaseq_val = pd.read_table(val_file, index_col=0, header=0, compression = 'xz')


# In[4]:


# Output files
dim_train_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata", subsample_dataset, "train_tune_dim.pickle")
dim_val_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata", subsample_dataset, "validation_tune_dim.pickle")


# In[5]:


rnaseq_train.head()


# In[6]:


rnaseq_val.head()


# In[7]:


# Save shape of input dataset
dim_train= rnaseq_train.shape
dim_val= rnaseq_val.shape

with open(dim_train_file, 'wb') as f:
    pickle.dump(dim_train, f)

del rnaseq_train

with open(dim_val_file, 'wb') as f:
    pickle.dump(dim_val, f)

del rnaseq_val

