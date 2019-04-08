
# coding: utf-8

# # Parse LINCS L1000 metadata
# 
# **By Alexandra Lee** 
# 
# **created January 2019 **
# 
# Parse metadata to determine if there are sufficient samples with multiple dose concentrations to train autencoder
# 
# Also explore the metadata to determine the breakdown of different cancer types, tissue types, drug types that exist in order to determine the scope of the analysis

# In[ ]:


import pandas as pd
import os
import numpy as np

import sys
from cmapPy.pandasGEXpress.parse import parse

randomState = 123
from numpy.random import seed
seed(randomState)


# In[2]:


# Load arguments
metadata_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata","GSE92742_Broad_LINCS_inst_info.txt")
cell_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata","GSE92742_Broad_LINCS_cell_info.txt")


# In[3]:


# Read sample metadata
metadata = pd.read_table(metadata_file, index_col=None, dtype=str)
metadata.head(10)


# In[4]:


# Read cell line metadata
cell_line_metadata = pd.read_table(cell_file, index_col=None, dtype=str)
cell_line_metadata.head(10)


# In[5]:


# Merge sample metadata and cell line metadata
metadata = metadata.merge(cell_line_metadata, on='cell_id', how='inner')
metadata.head(10)


# In[6]:


# Get unique pairs of (drug, cell line)
drug_cell_line_pairs = (
    metadata
    .groupby(['pert_iname', 'pert_type', 'cell_id', 'sample_type', 'primary_site', 'subtype'])
    .size()
    .reset_index()
    .rename(columns={0:'count'})
)

drug_cell_line_pairs.head(5)


# In[7]:


get_ipython().run_cell_magic('time', '', "# Filter by dose concentration and time points\nnum_pairs = drug_cell_line_pairs.shape[0]\n\nmultiple_dose_conc = pd.DataFrame(columns=['drug',\n                                           'drug type',\n                                           'cell line',\n                                           'sample type', \n                                           'primary site',\n                                           'subtype',\n                                           'time point', \n                                           'drug dose',\n                                           'count'])\n\nfor index, row in drug_cell_line_pairs.iterrows():\n    \n    # Select samples with specific drug and cell line\n    drug, drug_type, cell_line, sample_type, primary_site, cancer_type = row['pert_iname'],row['pert_type'], row['cell_id'], row['sample_type'], row['primary_site'], row['subtype']\n    selected_samples = metadata.query('pert_iname == @drug & cell_id == @cell_line', inplace=False)\n    \n    # Group samples by time point\n    timept_counts = (\n        selected_samples\n        .groupby(['pert_time'])\n        .size()\n        .reset_index()\n        .rename(columns={0:'count'})\n    )\n    \n    # For each time point group determine if multiple dose concentrations were measured\n    for timept in timept_counts['pert_time']:\n        samples_per_timept = selected_samples.query('pert_time == @timept', inplace=False)   \n        \n        # Get counts for the different dose concentrations\n        dose_conc_counts = (\n            samples_per_timept\n            .groupby(['pert_dose'])\n            .size()\n            .reset_index()\n            .rename(columns={0:'count'})\n        )\n\n        # Keep track of how many samples have multiple dose concentrations\n        num_dose_conc = dose_conc_counts.shape[0]\n        \n        if num_dose_conc > 1:\n            \n            for index,row in dose_conc_counts.iterrows():\n                dose_conc, count = row['pert_dose'], row['count']\n                multiple_dose_conc = multiple_dose_conc.append({'drug':drug,\n                                                                'drug type': drug_type,\n                                                                'cell line':cell_line,\n                                                                'sample type': sample_type,\n                                                                'primary site': primary_site,\n                                                                'subtype': cancer_type,\n                                                                'time point': timept,\n                                                                'drug dose':dose_conc,\n                                                                'count': count},\n                                                               ignore_index=True)")


# In[8]:


multiple_dose_conc.head(10)


# In[9]:


# Get the number of samples that have multiple dose concentrations
multiple_dose_conc['count'].sum()


# In[10]:


# Get the number of conditions (same drug, cell line, time point with multiple drug dose concentrations)
conditions_count = (
    multiple_dose_conc
    .groupby(['drug', 'drug type', 'cell line', 'sample type', 'primary site', 'subtype', 'time point'])
    .size()
    .reset_index()
    .rename(columns={0:'number of dose concentrations'})
)

conditions_count.shape


# In[11]:


conditions_count.head(20)


# In[12]:


# What is the counts for the different subtypes
top_subtype = conditions_count['subtype'].value_counts().index[0]
print(top_subtype)
conditions_count['subtype'].value_counts()


# In[13]:


# What is the counts for the different drugs
top_drug = conditions_count['drug'].value_counts().index[0]
print(top_drug)
conditions_count['drug'].value_counts()


# In[26]:


# What is the counts for the different primary tissue site
top_tissue = conditions_count['primary site'].value_counts().index[0]
print(top_tissue)
conditions_count['primary site'].value_counts()


# In[27]:


# Filter by the top primary site (tissue types)
conditions_count[conditions_count['primary site'] == top_tissue]['subtype'].value_counts()


# In[28]:


conditions_count[(conditions_count['primary site'] == top_tissue) &
                 (conditions_count['subtype'] == top_subtype) &
                 (conditions_count['time point'] == '24')]


# In[41]:


conditions_count[(conditions_count['primary site'] == top_tissue) &
                 (conditions_count['subtype'] == top_subtype) &
                 (conditions_count['time point'] == '24')]['drug type'].value_counts()


# In[17]:


multiple_dose_conc[(multiple_dose_conc['primary site'] == top_tissue) &
                 (multiple_dose_conc['subtype'] == top_subtype) &
                 (multiple_dose_conc['time point'] == '24')]


# In[33]:


top_drug = multiple_dose_conc[(multiple_dose_conc['primary site'] == top_tissue) &
                 (multiple_dose_conc['subtype'] == top_subtype) &
                 (multiple_dose_conc['time point'] == '24')]['drug'].value_counts().index[0]
print(top_drug)


# In[35]:


multiple_dose_conc[(multiple_dose_conc['primary site'] == top_tissue) &
                 (multiple_dose_conc['subtype'] == top_subtype) &
                 (multiple_dose_conc['time point'] == '24') &
                  (multiple_dose_conc['drug'] == top_drug)]['count'].sum()


# In[38]:


multiple_dose_conc[(multiple_dose_conc['primary site'] == top_tissue) &
                 (multiple_dose_conc['subtype'] == top_subtype) &
                 (multiple_dose_conc['time point'] == '24')]['count'].sum()


# In[19]:


conditions_count[conditions_count['primary site'] == top_tissue]['cell line'].value_counts()


# In[20]:


conditions_count['number of dose concentrations'].max()


# In[21]:


conditions_count.loc[conditions_count['number of dose concentrations'] == 44]


# In[22]:


# Samples from first condition with 44 doses
drug_name = 'vorinostat'
cell_line_name = 'MCF7'
time_pt = '24'
multiple_dose_conc.loc[(multiple_dose_conc['drug'] == drug_name) &
                       (multiple_dose_conc['cell line'] == cell_line_name) &
                       (multiple_dose_conc['time point'] == time_pt)]['count'].sum()


# In[23]:


drug_name = 'vorinostat'
cell_line_name = 'PC3'
time_pt = '24'
multiple_dose_conc.loc[(multiple_dose_conc['drug'] == drug_name) &
                       (multiple_dose_conc['cell line'] == cell_line_name) &
                       (multiple_dose_conc['time point'] == time_pt)]['count'].sum()


# In[24]:


# Output
dose_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata","multiple_dose_conc_counts.txt")
conditions_file = os.path.join(
    os.path.dirname(
        os.getcwd()), "metadata","conditions_counts.txt")

multiple_dose_conc.to_csv(dose_file, sep='\t')
conditions_count.to_csv(conditions_file, sep='\t')

