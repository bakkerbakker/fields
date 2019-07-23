# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:48:35 2019

@author: jesse bakker (bakke557@umn.edu)
"""

# Utility functions

# import libraries
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure
from fields import *

# Display stats for raster array
def band_array_stats(band_array):
    '''Prints some basic stats (min, max, mean, median)
    foe each band in the band array, returns the input array'''
    
    # define empty list to append band stats into
    band_stats = []
    
    # band order in demo tif: blue, green, red, NIR 
    for band in band_array:
        band_stats.append({
            'min': band.min(),
            'max': band.max(),
            'mean': band.mean(),
            'median': np.median(band)
        })
    
    print(band_stats)
    
    return band_array   

# Transpose array
def transpose_3NM_to_NM3(input_array):
    '''
    Transposes a (3, N, M) array to (N, M, 3) array for passing mask
    to multi-band array
    '''
    print("input array shape: ", input_array.shape)
    transposed_array = np.transpose(input_array, (1, 2, 0))
    print("transposed array shape: ", transposed_array.shape)    
    return transposed_array

# Normalize array values to be passed to processing functions
def normalize(array):
    '''normalized numpy array into scale 0.0 - 1.0'''

    return ((array - array.min())/(array.max() - array.min()))  

def visualize_2D_array_0_1000(array, title = '', cmap = 'viridis'):
    '''
    Plot 2D array, subset to [0:1000, 0:1000]
    '''
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(array[0:1000, 0:1000], cmap = cmap)
    
def visualize_3_band_image_array_0_1000(input_3_band_array, title = ''):
    '''
    Plot 3-channel image array, ex: rgb or false color.
    '''
    fig = plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(input_3_band_array[0:1000, 0:1000])
    
def clip_array_abs(array, min_val, max_val):
    band_clp = np.clip((array), a_min = min_val, a_max = max_val)
    return band_clp
  