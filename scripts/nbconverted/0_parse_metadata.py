
# coding: utf-8

# In[2]:


# -----------------------------------------------------------------------------------------------------------------------
# Alexandra Lee 
# (created January 2019) 
#
# Parse metadata to determine if there are sufficient samples with multiple 
# dose concentrations to train autencoder
# -------------------------------------------------------------------------------------------------------------------
import pandas as pd
import os
import numpy as np

import sys
from cmapPy.pandasGEXpress.parse import parse

randomState = 123
from numpy.random import seed
seed(randomState)


# In[3]:


# Load arguments
metadata_file = os.path.join(os.path.dirname(os.getcwd()), "metadata","GSE92742_Broad_LINCS_inst_info.txt")


# In[7]:


# Read metadata
metadata = pd.read_table(metadata_file, index_col=0, dtype=str)
metadata.head(10)


# In[4]:


# Get unique pairs of (drug, cell line)
drug_cell_line_pairs = (
    metadata
    .groupby(['pert_iname','cell_id'])
    .size()
    .reset_index()
    .rename(columns={0:'count'})
)

drug_cell_line_pairs.head(5)


# In[5]:


get_ipython().run_cell_magic('time', '', "# Filter by dose concentration and time points\nnum_pairs = drug_cell_line_pairs.shape[0]\n\nmultiple_dose_conc = pd.DataFrame(columns=['drug', 'cell line', 'time point', 'drug dose', 'count'])\n\nfor index, row in drug_cell_line_pairs.iterrows():\n    \n    # Select samples with specific drug and cell line\n    drug, cell_line = row['pert_iname'], row['cell_id']\n    selected_samples = metadata.query('pert_iname == @drug & cell_id == @cell_line', inplace=False)\n    \n    # Group samples by time point\n    timept_counts = (\n        selected_samples\n        .groupby(['pert_time'])\n        .size()\n        .reset_index()\n        .rename(columns={0:'count'})\n    )\n    \n    # For each time point group determine if multiple dose concentrations were measured\n    for timept in timept_counts['pert_time']:\n        samples_per_timept = selected_samples.query('pert_time == @timept', inplace=False)   \n        \n        # Get counts for the different dose concentrations\n        dose_conc_counts = (\n            samples_per_timept\n            .groupby(['pert_dose'])\n            .size()\n            .reset_index()\n            .rename(columns={0:'count'})\n        )\n\n        # Keep track of how many samples have multiple dose concentrations\n        num_dose_conc = dose_conc_counts.shape[0]\n        \n        if num_dose_conc > 1:\n            \n            for index,row in dose_conc_counts.iterrows():\n                dose_conc, count = row['pert_dose'], row['count']\n                multiple_dose_conc = multiple_dose_conc.append({'drug':drug,\n                                                                'cell line':cell_line,\n                                                                'time point': timept,\n                                                                'drug dose':dose_conc,\n                                                                'count': count},\n                                                               ignore_index=True)")


# In[14]:


multiple_dose_conc.head(10)


# In[21]:


# Get the number of samples that have multiple dose concentrations
multiple_dose_conc['count'].sum()


# In[23]:


# Get the number of conditions (same drug, cell line, time point with multiple drug dose concentrations)
conditions_count = (
    multiple_dose_conc
    .groupby(['drug', 'cell line', 'time point'])
    .size()
    .reset_index()
    .rename(columns={0:'number of dose concentrations'})
)

conditions_count.shape


# In[24]:


conditions_count.head(20)


# In[25]:


conditions_count['number of dose concentrations'].max()


# In[34]:


conditions_count.loc[conditions_count['number of dose concentrations'] == 44]


# In[49]:


# Samples from first condition with 44 doses
drug_name = 'vorinostat'
cell_line_name = 'MCF7'
time_pt = '24'
multiple_dose_conc.loc[(multiple_dose_conc['drug'] == drug_name) &
                       (multiple_dose_conc['cell line'] == cell_line_name) &
                       (multiple_dose_conc['time point'] == time_pt)]['count'].sum()


# In[50]:


drug_name = 'vorinostat'
cell_line_name = 'PC3'
time_pt = '24'
multiple_dose_conc.loc[(multiple_dose_conc['drug'] == drug_name) &
                       (multiple_dose_conc['cell line'] == cell_line_name) &
                       (multiple_dose_conc['time point'] == time_pt)]['count'].sum()


# In[20]:


# Output
dose_file = os.path.join(os.path.dirname(os.getcwd()), "metadata","multiple_dose_conc_counts.txt")
conditions_file = os.path.join(os.path.dirname(os.getcwd()), "metadata","conditions_counts.txt")

multiple_dose_conc.to_csv(dose_file, sep='\t')
conditions_count.to_csv(conditions_file, sep='\t')

