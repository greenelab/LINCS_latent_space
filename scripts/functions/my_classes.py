import pandas as pd
import numpy as np
import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_file, chunk_size, num_samples):
        'Initialization'
        self.chunk_size = chunk_size
        self.data_file = data_file
        self.num_samples = num_samples
        self.data = pd.read_table(
            self.data_file,
            chunksize=self.chunk_size,
            index_col=0)

    # Reinitialize (open) data file to cycle through for the next epoch
    def _reinitialize_data(self):
        #print("I reinitalized the data for you")
        self.data = pd.read_table(
            self.data_file,
            chunksize=self.chunk_size,
            index_col=0)

    # Each call requests a batch index between 0 and the total number of batches
    def __len__(self):
        'Denotes the number of batches per epoch'
        num_chunks = int(np.floor(self.num_samples / self.chunk_size))
        #print('return {} batches per epoch'.format(num_chunks))
        return num_chunks

    # Returns a batch of the data
    def __getitem__(self, index):
        try:
            while True:
                #print("Returning index: {}.".format(index))
                data_chunk = self.data.__next__()

                return (data_chunk, None)
        except StopIteration:
            pass

    # Triggered once at the very beginning as well as at the end of each epoch
    # Shuffling the order in which examples are fed to the classifier is helpful
    # so that batches between epochs do not look alike. Doing so will eventually make our model more robust.
    def on_epoch_end(self):
        #print("I ended the epoch")
        self._reinitialize_data()
