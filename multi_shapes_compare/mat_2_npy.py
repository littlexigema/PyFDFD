import numpy as np
import scipy.io
import os

def func(data_dir):
    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            mat_filepath = os.path.join(data_dir, filename)
            mat_contents = scipy.io.loadmat(mat_filepath)
            # Assuming the .mat file contains only one variable
            variable_name = list(mat_contents.keys())[-1]
            variable_value = mat_contents[variable_name]
            npy_filepath = os.path.join(data_dir, filename.replace('.mat', '.npy'))
            np.save(npy_filepath, variable_value)

# Example usage
func('./BPS')
