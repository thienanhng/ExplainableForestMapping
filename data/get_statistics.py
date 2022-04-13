""" 
Compute statistics for a set of files:
    - mean and std of pixel values for image files
    - class frequency for target files
"""

import os
from tqdm import tqdm
import csv
import rasterio
import numpy as np
from math import ceil

project_dir =  os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) 

def get_fn_list(csv_fn):
    with open(csv_fn, 'r') as f_csv:
        reader = csv.reader(f_csv)
        fn_list = list(reader)
    return [fn[0] for fn in fn_list]

def get_image_statistics(fn_list, p_pix):
    """
    Computes the mean and standard deviation of the images listed in fn_list.

    Args:
        - fn_list (list of str): list of paths to the files to use to calculate the statistics
        - p_pix (float): proportion of random pixels to use to calculate the statistics.

    """

    print('Sampling the dataset')
    sampled_pix = []
    n_total_pix = 0
    for fn in tqdm(fn_list):
        if os.path.splitext(fn)[-1] == '.tif':
            with rasterio.open(fn, 'r') as f:
                data = f.read()
                nodata_val = f.nodata
            data = data.reshape(data.shape[0], -1)
            n_total_pix += data.shape[-1]
            n_pix = ceil(p_pix * data.shape[-1])
            if n_pix > 0:
                idxs = np.random.choice(data.shape[-1], size=n_pix)
                data = data[:,idxs]
                sampled_pix.append(data[:,np.any(data != nodata_val, axis=0)])

    sampled_pix = np.concatenate(sampled_pix, axis=1)
    print('{:.1e} %  pixels sampled in total'.format(sampled_pix.shape[-1]/n_total_pix))

    print('Computing the mean')
    mean = sampled_pix.mean(axis=-1, dtype=np.float64)
    print(mean)

    print('Computing the std')
    std = sampled_pix.std(axis=-1, dtype=np.float64)
    print(std)

def get_target_statistics(fn_list, p_pix, n_classes, non_empty_only = False):
    """
    Computes the frequency of each class in a target dataset (useful to compute 
    loss weights)

    Args:
        - fn_list (list of str): list of paths to the files to use to calculate the statistics
        - p_pix (float): proportion of random pixels to use to calculate the statistics
        - n_classes (int): number of classes (numbered from 0 to n_classes - 1 in the provided files)
        - non_empty_only (bool): whether or not to use only files that contain at least one pixel from a class other
            than class 0 
    """
    sampled_pix = []
    frequencies = [0] * n_classes
    n_files = 0
    for fn in tqdm(fn_list):
        if os.path.splitext(fn)[-1] == '.tif':
            with rasterio.open(fn, 'r') as f:
                data = f.read()
            data = data.reshape(-1)
            n_pix = ceil(p_pix * data.shape[0])
            if n_pix > 0:
                idxs = np.random.choice(data.shape[0], size=n_pix)
                sampled_pix.append(data[idxs])
                unique, counts = np.unique(data[idxs], return_counts=True)
                if not(non_empty_only) or (non_empty_only and (unique != 0).any()): 
                    n_files += 1
                    for i, c in enumerate(unique):
                        frequencies[c] += counts[i]/n_pix
    frequencies = [f/n_files for f in frequencies]
    print(frequencies)


if __name__ == "__main__":
    
    ### image statistics
    # source = 'TH'
    # train_fn_list_fn = os.path.join('data', 'csv', source + '_train.csv')
    # val_fn_list_fn = os.path.join('data', 'csv', source + '_val.csv')
    # train_fn_list = get_fn_list(train_fn_list_fn)
    # val_fn_list = get_fn_list(val_fn_list_fn)
    # p_pix = 1e-5 # proportion of pixels used per image
    # get_image_statistics(train_fn_list + val_fn_list, p_pix)
    
    ### target statistics
    fn_list = get_fn_list(os.path.join(project_dir, 'data/csv/TLM5c_test.csv'))
    get_target_statistics(fn_list, p_pix = 1, n_classes = 5, non_empty_only=False)
