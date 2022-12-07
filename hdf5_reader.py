import os
import h5py
import json
import glob
import contextlib
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
from contextlib import contextmanager
from tensorflow import keras
from IPython.display import HTML, display, Markdown
from skimage.transform import resize


@contextmanager
def hdf5_open_file(hdf5_filename):
    """Open the .spikes.hdf5 file and return handle to file.

    :param hdf5_filename: [str] Full path to .spikes.hdf5 file to read.
    :return hdf5_file: [object] Handle of file.
    """
    hdf5_file = None

    try:
        hdf5_file = tb.open_file(hdf5_filename, 'r')
        yield hdf5_file
    finally:
        if hdf5_file is not None:
            hdf5_file.flush()
            hdf5_file.close()


def hdf5_read_table(hdf5_filename, table_name):
    with hdf5_open_file(hdf5_filename) as hdf5_file:
        # Check that array exists
        if not hdf5_file.__contains__(f'/{table_name}'):
            raise IOError(f'\'{table_name}\' not present in \'{hdf5_filename}\'')

        # Read data and return
        hdf5_table = hdf5_file.get_node('/', f'{table_name}')
        data = hdf5_table[:]
        # Convert to pandas DataFrame
        data = pd.DataFrame.from_records(data)

    # Convert column type
    if table_name == 'metadata':
        columns_to_convert = data.select_dtypes([np.object]).columns
    else:
        columns_to_convert = list()

    if len(columns_to_convert) > 0:
        for col in columns_to_convert:
            data[col] = data[col].str.decode('utf-8')
			
    return data


def hdf5_read_array(hdf5_filename, array_name):
    # Open file for reading
    with hdf5_open_file(hdf5_filename) as hdf5_file:
        # Check that array exists
        if not hdf5_file.__contains__('/%s' % array_name):
            raise IOError('\'%s\' not present in \'%s\'' % (array_name, hdf5_filename))

        # Read data and return
        hdf5_array = hdf5_file.get_node('/', '%s' % array_name)
        data = hdf5_array[:]

    return data

###################################################################

# # sanity check: load metadata
# data_base_name = 'df_prediction_behavior_naive_lfp'
# data_dir = '/content/drive/MyDrive/data_ephys_predictions'
# data_path = os.path.join(data_dir, f'{data_base_name}_RS22.hdf5')

# metadata_df = hdf5_read_table(data_path, 'metadata')
# from IPython.display import display
# display(metadata_df)

# # Read time series data from dataframe (this function reads one signal at a time)
# a_name = f"{metadata_df.iloc[0]['array_name']}_{metadata_df.iloc[0]['animal']}"
# signal_stim = hdf5_read_array(data_path, a_name)

# # Now you've read the signal :)
# plt.plot(signal_stim)