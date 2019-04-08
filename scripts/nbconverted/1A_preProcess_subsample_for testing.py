
# coding: utf-8

# # Pre-process LINCS L1000 dataset
# 
# Pre-processing steps include:
#     
# 1. Normalize data
# 2. Partition dataset into training and validation sets
# 
# Note:  Using python 2 in order to support parsing cmap function

# In[ ]:


import pandas as pd
import os
import numpy as np
from scipy.stats import variation
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt

import sys
from cmapPy.pandasGEXpress.parse import parse

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Output files
train_file = "/home/alexandra/Documents/Data/LINCS_tuning/train_model_input.txt.xz"
validation_file = "/home/alexandra/Documents/Data/LINCS_tuning/validation_model_input.txt.xz"


# ## About the data
# Read in gene expression data (GCToo object with 3 embedded dataframes include data_df)
# Data downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742
#  
# cid = samples
# 
# rid = genes
# 
# values = normalized and imputed (based on landmark genes) gene expression --> log fold change compared against negative control
# 
# Note:  Data is too large to be housed in repo so instead it is housed on local desktop

# In[3]:


get_ipython().run_cell_magic('time', '', 'data_file = "/home/alexandra/Documents/Data/LINCS/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"\n\n# Keep only landmark genes\ngene_info_file = os.path.join(\n    os.path.dirname(\n        os.getcwd()), "metadata","GSE92742_Broad_LINCS_gene_info.txt")\n\ngene_info = pd.read_table(gene_info_file, dtype=str)\nlandmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]\n\ndata = parse(data_file, rid = landmark_gene_row_ids)\n\ndata_df = data.data_df.T')


# In[4]:


data_df.shape


# In[5]:


# Normalization
# scale data to range (0,1) per gene
data_scaled_df = (
    preprocessing
    .MinMaxScaler()
    .fit_transform(data_df)
)

data_scaled_df = pd.DataFrame(data_scaled_df,
                                columns=data_df.columns,
                                index=data_df.index)
del data_df

data_scaled_df.head(5)
print(data_scaled_df.shape)


# In[6]:


sns.distplot(data_scaled_df.iloc[5])


# In[7]:


# Subsample dataset in order to tune parameters
subsample_frac = 0.01
subsample_data = data_scaled_df.sample(frac=subsample_frac, random_state=randomState)
print(subsample_data.shape)


# In[8]:


# Split dataset into training and validation sets
validation_frac = 0.2
validation_df = subsample_data.sample(frac=validation_frac, random_state=randomState)
train_df = subsample_data.drop(validation_df.index)

print(validation_df.shape)
print(train_df.shape)


# In[9]:


# Output
train_df.to_csv(train_file, sep='\t', compression='xz')
validation_df.to_csv(validation_file, sep='\t', compression='xz')

