
# coding: utf-8

# In[1]:


# -----------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee
# (created December 2018) 
#
# Get dimension of feature space for input into 
# 3_tybalt_2layer_model_generator.py to train autoencoder
# 
# --------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
import pickle

randomState = 123


# In[2]:


get_ipython().run_cell_magic('time', '', '# Load dataset\ndata_file =  "/home/alexandra/Documents/Data/LINCS/validation_model_input.txt.xz"\n\nrnaseq = pd.read_table(data_file, sep=\'\\t\', index_col=0, header=0, compression = \'xz\')\n\n# output\ndim_file = os.path.join(os.path.dirname(os.getcwd()), "metadata", "validation_dim.pickle")')


# In[3]:


rnaseq.head(5)


# In[4]:


# Save shape of input dataset
dim= rnaseq.shape

with open(dim_file, 'wb') as f:
    pickle.dump(dim, f)

del rnaseq, data_file, dim_file

