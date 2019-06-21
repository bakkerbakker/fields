# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:27:59 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### field Read Raster functions

# Import libraries
import rasterio
import numpy as np

# Read raster from stacked tif
def read_tif_to_array (filename):
    '''Opens the input tif file using raster.io and converts
    it to a multi-band np array. Returns the array.'''
    
    # band order in demo tif: blue, green, red, NIR 
    with rasterio.open(filename) as src:
        band_array = src.read()
        band_array = np.array(band_array)
    
    print("imported raster array shape: ", band_array.shape)
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
