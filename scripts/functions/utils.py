# -----------------------------------------------------------------------------------------------------------------------
# By Alexandra Lee
# (updated December 2018)
#
# Get dimension of feature space for input into Tybalt training
# --------------------------------------------------------------------------------------------------------------------
import os
import pandas as pd
#import pickle

randomState = 123


def get_dim_input(data_dir, train_or_test):
    """
    Read in gene expression dataset and return dimensions
    """

    # Load dataset
    data_file = os.path.join(data_dir,
                             train_or_test+"_model_input.txt.xz")
    rnaseq = pd.read_table(data_file, sep='\t', index_col=0,
                           header=0, compression='xz')

    # output
    #dim_file = os.path.join(data_dir, "dataset_dim.pickle")

    # Save shape of input dataset
    dim_dataset = rnaseq.shape
    return(dim_dataset)
    # with open(dim_file, 'wb') as f:
    #    pickle.dump(dim_dataset, f)
