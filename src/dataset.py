import os
import numpy as np
import pandas as pd
from io import StringIO
import re
from functools import reduce

class SKADataset:
    """
    SKA dataset wrapper.
    """

    def __init__(self, train_set_path=None, test_set_path=None, subset=1.0):
    
        # Save training and test set paths
        self.train_set_path = train_set_path
        self.test_set_path = test_set_path

        # Save column names
        self.col_names = ['ID',
              'RA (core)',
              'DEC (core)',
              'RA (centroid)',
              'DEC (centroid)',
              'FLUX',
              'Core frac',
              'BMAJ',
              'BMIN',
              'PA',
              'SIZE',
              'CLASS',
              'SELECTION',
              'x',
              'y']
    
        # Process the training set
        self.raw_train_df = None
        if self.train_set_path is not None:
            assert os.path.exists(
                self.train_set_path
            ), "Missing SKA training set .txt file"
            self.train_df_path = f"{os.path.splitext(self.train_set_path)[0]}.pkl"
            self.raw_train_df = self._load_dataset(
                self.train_set_path, self.train_df_path
            )
            if subset < 1.0:
                self.raw_train_df = self._get_portion(self.raw_train_df, subset)

    def _load_dataset(self, dataset_path, dataframe_path):
        """
        Loads the SKA dataset into a Pandas DataFrame,
        starting from a specifically-formatted txt file.
        """
        # Load the DataFrame, if it was already pickled before
        if os.path.exists(dataframe_path):
            try:
                return pd.read_pickle(dataframe_path)
            except ValueError:
                pass

        # Otherwise load the txt file and extract data
        else:
            df = self._prepare_dataset(dataset_path)

        # Save the dataframe into a pickle file
        df.to_pickle(dataframe_path)
        
        return df


    def _get_portion(self, df, subset=1.0):
        """
        Returns a random subset of the whole dataframe.
        """
        amount = int(df.shape[0] * subset)
        random_indexes = np.random.choice(
            np.arange(df.shape[0]), size=amount, replace=False
        )
        return df.iloc[random_indexes].reset_index(drop=True)

    def _prepare_dataset(self, dataset_path):

        df = pd.read_csv(dataset_path, skiprows=18, header=None, names=self.col_names, delimiter=' ', skipinitialspace=True)

        return df
    
