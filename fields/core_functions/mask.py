# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:06:37 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Masking functions

# import libraries
import numpy as np
import numpy.ma as ma
import scipy.ndimage as ndi
from fields.core_functions.ndvi import compute_ndvi
from fields.IO.read_rasters import read_tif_to_array
from fields.core_functions.edges import edges_from_3Darray_max, edges_from_3Darray_sum, combine_edges
from fields.core_functions.ndvi import ndvi_range_from_stack
from fields.utilities.utilities import normalize


def return_masked_array(input_array, lower_percentile, upper_percentile):
    # get threshold values based on array percentiles
    lower_thresh = np.percentile(input_array, lower_percentile, interpolation = 'lower')
    print("Lower threshold:", lower_thresh, "at percentile:", lower_percentile)
    upper_thresh = np.percentile(input_array, upper_percentile, interpolation = 'higher')
    print("Upper threshold:", upper_thresh, "at percentile:", upper_percentile)
    
    masked_array = ma.masked_outside(input_array, lower_thresh, upper_thresh)
    
    return masked_array
  
def apply_mask(array_to_mask, mask_array, masked_value = np.nan):
    '''
    Set values covered by maskArray to False in the input array.
    '''
    array_to_mask[mask_array] = masked_value    
    
    return array_to_mask

def create_combined_mask(ndvi, edges, ndvi_weight=1, edges_weight=1):
    return (ndvi_weight*ndvi) + (edges_weight*edges)


def create_binary_mask(input_array, threshold = 0.15, fill_holes = True):
    binary_mask = np.zeros_like(input_array)
    binary_mask[normalize(input_array) < threshold] = 1
    if fill_holes == True:
        binary_mask = ndi.binary_fill_holes(binary_mask)
    return binary_mask

def create_ternary_mask(input_array, lower_thresh = 0.15, upper_thresh = 0.8):
    ternary_mask = np.zeros_like(input_array)
    ternary_mask[normalize(input_array) < lower_thresh] = 1
    ternary_mask[normalize(input_array) > upper_thresh] = 2

    return ternary_mask


### *ERIC* how do we restructure this loop to be modular and efficient with IO???
### Need to figure out to work with the loop for combining with other tiles
def time_stack_ndvi_and_edges(folder_path, rasters):
    '''
    function to read in a list of raster files, calculate NDVI for each raster
    and edges for each raster band. It returns a stack of NDVI arrays and 
    cumulative arrays for Edge Max and Edge Sum from all the bands.
    '''
    
    count = 0
    
    for raster in rasters:
        # define raster path
        filepath = folder_path + raster
        
        # read raster to array
        band_stack = read_tif_to_array(filepath)

        # compute NDVI
        ndvi_array = compute_ndvi(band_stack[3], band_stack[2])

        # add NDVI to the band stack to process through the edge detection step along with the other bands
        band_stack = np.concatenate((band_stack, ndvi_array[np.newaxis,:,:]), axis=0)

        # calculate edges from band stack
        edges_max = edges_from_3Darray_max(band_stack)
        edges_sum = edges_from_3Darray_sum(band_stack)
        
        if count == 0:
            # create NDVI stack
            ndvi_stack = np.copy(ndvi_array)
            print("ndvi array shape:", ndvi_stack.shape)
            # create array for cumulative edges
            cumulative_edges_max = np.copy(edges_max)
            cumulative_edges_sum = np.copy(edges_sum)
        else:
            # append NDVI bands into NDVI stack
            ndvi_stack = np.dstack((ndvi_stack, ndvi_array))
            print("ndvi array shape:", ndvi_stack.shape)
            # compute cumulative edges into single array
            cumulative_edges_max = np.maximum(cumulative_edges_max, edges_max)
            cumulative_edges_sum += edges_sum
           
        # clear variables that aren't needed in next steps
        band_stack = None
        ndvi_array = None
        edges_max = None
        edges_sum = None
        count += 1
        
    return ndvi_stack, cumulative_edges_max, cumulative_edges_sum

#### Combined functions for workflow
def mask_func(**mask_inputs):
   folder_path = mask_inputs['folder_path']
   rasters = mask_inputs['rasters']
   max_weight = mask_inputs['max_edge_weight']
   sum_weight = mask_inputs['sum_edge_weight']
   ndvi_weight = mask_inputs['ndvi_weight']
   edges_weight = mask_inputs['edges_weight']
   binary_threshold = mask_inputs['binary_threshold']
   
   # Loop through rasters to build a stacked array of ndvi and two cumulative edge arrays: max and sum
   ndvi_stack, edges_max, edges_sum = time_stack_ndvi_and_edges(folder_path, rasters)
   # combine edge arrays into a single normalized (0-1) array, each weighted by an input factor
   edges_combined = normalize(combine_edges(normalize(edges_max), normalize(edges_sum), 
                                         max_weight=max_weight, sum_weight=sum_weight))
   # combine edge array with normalized (0-1) ndvi range, each weighted by an input factor
   mask_combo = create_combined_mask(ndvi = 1 - normalize(ndvi_range_from_stack(ndvi_stack)), 
                                     edges = edges_combined, 
                                     ndvi_weight = ndvi_weight, edges_weight = edges_weight)
   # create a binary mask (crop/non-crop) to use during segmentation
   binary_mask = create_binary_mask(mask_combo, threshold=binary_threshold, fill_holes=True)
   
   return binary_mask
        
