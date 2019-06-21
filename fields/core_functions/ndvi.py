# -*- coding: utf-8 -*-
"""
Created on Thu May 30 11:51:33 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### NDVI processing functions

# import libraries
import numpy as np

# Compute NDVI
def compute_ndvi(nir, red):
    '''
    computes the Normalized Difference Vegetation Index
    '''
    np.seterr(divide='ignore', invalid='ignore')
    nir = nir.astype(float)
    red = red.astype(float)
    return ((nir - red)/(nir + red))

def ndvi_range_from_stack(ndvi_stack):
    '''
    Returns an array of the range of values for each cell in a stacked array
    '''
    return np.amax(ndvi_stack, axis = 2) - np.amin(ndvi_stack, axis = 2)

def ndvi_min_from_stack(ndvi_stack):
    return np.amin(ndvi_stack, axis = 2)

def ndvi_max_from_stack(ndvi_stack):
    return np.amax(ndvi_stack, axis = 2)

#
#
#def ndvi_stack_in_loop(ndvi_in, count):
#    count = count
#    if count == 0:
#        # create NDVI stack
#        ndvi_stack = np.copy(ndvi_in)
#        print("ndvi array shape:", ndvi_stack.shape)
#    else:
#        # append NDVI bands into NDVI stack
#        ndvi_stack = np.dstack((ndvi_stack, ndvi_in))
#        print("ndvi array shape:", ndvi_stack.shape)
#    return ndvi_stack
#
#def ndvi_timestack(raster_list):
#    count = 0 
#    for raster in raster_list:
#        bands_bgrn = read_tif_to_array(raster)
#      
#        if count == 0:
#            ndvi_stack = compute_ndvi(bands_bgrn[3], bands_bgrn[2])
#        else:
#            ndvi = compute_ndvi(bands_bgrn[3], bands_bgrn[2])
#            ndvi_stack = np.dstack((ndvi_stack, ndvi))
#        count += 1
#      
#    return ndvi_stack