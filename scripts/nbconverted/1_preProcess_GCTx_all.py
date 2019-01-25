
# coding: utf-8

# In[3]:


# -----------------------------------------------------------------------------------------------------------------------
# Alexandra Lee 
# (created October 2018) 
#
# Pre-process LINCS L1000 dataset:
# 1. Normalize data
# 2. Partition dataset into training and validation sets
#
# Note:  Using python 2 in order to support parsing cmap function
# -------------------------------------------------------------------------------------------------------------------
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


# In[4]:


# Output files
train_file = "/home/alexandra/Documents/Data/LINCS/train_model_input.txt.xz"
validation_file = "/home/alexandra/Documents/Data/LINCS/validation_model_input.txt.xz"


# In[5]:


get_ipython().run_cell_magic('time', '', '# Read in gene expression data (GCToo object with 3 embedded dataframes include data_df)\n# Data downloaded from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742\n# \n# cid = samples\n# rid = genes\n# values = normalized and imputed (based on landmark genes) gene expression --> log fold change compared against\n#          negative control\n\n# Note:  Data is too large to be housed in repo\ndata_file = "/home/alexandra/Documents/Data/LINCS/GSE92742_Broad_LINCS_Level3_INF_mlr12k_n1319138x12328.gctx"\n\n# Keep only landmark genes\ngene_info_file = os.path.join(os.path.dirname(os.getcwd()), "metadata","GSE92742_Broad_LINCS_gene_info.txt")\ngene_info = pd.read_table(gene_info_file, dtype=str)\nlandmark_gene_row_ids = gene_info["pr_gene_id"][gene_info["pr_is_lm"] == "1"]\n\ndata = parse(data_file, rid = landmark_gene_row_ids)\n\ndata_df = data.data_df.T')


# In[4]:


data_df.shape


# In[6]:


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


# In[16]:


sns.distplot(data_scaled_df.iloc[5])


# In[6]:


# Split dataset into training and validation sets
validation_set_percent = 0.001
validation_df = data_scaled_df.sample(frac=validation_set_percent, random_state=randomState)
train_df = data_scaled_df.drop(validation_df.index)

print(validation_df.shape)
print(train_df.shape)


# In[7]:


# Output
#train_df.to_csv(train_file, sep='\t', compression='xz')
#validation_df.to_csv(validation_file, sep='\t', compression='xz')

