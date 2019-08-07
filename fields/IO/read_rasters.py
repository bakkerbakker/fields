# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:27:59 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### field Read Raster functions

# Import libraries
import rasterio
import numpy as np
from skimage import exposure
import dask.array as da
from fields.utilities.utilities import normalize, clip_array_abs

from fields import *

# Read raster from stacked tif
def read_tif_to_array (filename):
    '''Opens the input tif file using raster.io and converts
    it to a multi-band np array. Returns the array.'''
    
    # band order in demo tif: blue, green, red, NIR 
    with rasterio.open(filename) as src:
        band_array = src.read()
        band_array = np.array(band_array)
        band_array = da.from_array(band_array, chunks = (-1, 100, 100))
        
    
    print("imported raster array shape: ", band_array.shape, "array type:", type(band_array))
    return band_array

  
def select_bands_bgrn_to_rgb(band_array_bgrn):
    '''this function takes a four band 3D array in the band order BGRN
    and returns a 3D array of three bands in the order RGB
    for true color display'''
    
    raster_r = band_array_bgrn[2]
    raster_g = band_array_bgrn[1]
    raster_b = band_array_bgrn[0]
    
    band_array_rgb = np.array([raster_r,raster_g,raster_b])
    
    print("3D array of R, G, B bands")
    print("reordered raster array shape:", band_array_rgb.shape)
    
    return band_array_rgb


def prep_rgb_image(input_bgrn_array, gamma=1, clip_val_r = 1500, clip_val_g = 1500, clip_val_b = 1500): 
    '''
    Uses skimage exposure.adjust_gamma to make image more readable.
    '''
    # rearrange to rgb 3-band image
    rgb_input = np.array([exposure.adjust_gamma(normalize(clip_array_abs(input_bgrn_array[2], 0, clip_val_r)), gamma=gamma),
                          exposure.adjust_gamma(normalize(clip_array_abs(input_bgrn_array[1], 0, clip_val_g)), gamma=gamma),
                          exposure.adjust_gamma(normalize(clip_array_abs(input_bgrn_array[0], 0, clip_val_b)), gamma=gamma)])
    rgb_input = np.transpose(rgb_input, (1, 2, 0))
    print(rgb_input.shape)
    return rgb_input

def rgb_img_func(**rgb_img_inputs):
    # inputs 
    folder_path = rgb_img_inputs['folder_path']
    rasters = rgb_img_inputs['rasters']
    raster_image_index = rgb_img_inputs['raster_image_index']
    gamma=rgb_img_inputs['gamma']
    clip_val_r = rgb_img_inputs['clip_val_r'] 
    clip_val_g = rgb_img_inputs['clip_val_g'] 
    clip_val_b = rgb_img_inputs['clip_val_b']
    
    # read RGB from tif raster
    raster_fp = folder_path + rasters[raster_image_index]
    print(raster_fp)
    rgb_image = prep_rgb_image(read_tif_to_array(raster_fp), 
                               gamma=gamma, 
                               clip_val_r = clip_val_r, 
                               clip_val_g = clip_val_g, 
                               clip_val_b = clip_val_b)
    
    return rgb_image