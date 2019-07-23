# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:00:59 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Functions for finding edges

# import libraries
import numpy as np
from skimage import filters, exposure, feature




def edges_from_3Darray_max(array, perc=(0,99)):
  '''
  loops through each band in the array, 
  computes sobel edge value, and returns the 
  maximum edge value from all bands in all_edges array
  '''
  bands = np.arange(array.shape[0])
  # set a shape for the edge output to match the input
  shape = array[0].shape
  all_edges = np.zeros(shape, dtype=np.float)
  for band in bands:
    print("edge max band number:", band)
    # rescale band
    p_low, p_high = np.percentile(array[band], perc)
    edge_band = exposure.rescale_intensity(
            array[band],
            in_range=(p_low, p_high),
            out_range=(0, 255))

    # find edges for each band 
    edges = filters.sobel(edge_band)
    edges = exposure.rescale_intensity(edges, out_range=(0, 255))
    all_edges = np.maximum(all_edges, edges)
    
  all_edges = filters.gaussian(all_edges, 1)
  return all_edges    


def edges_from_3Darray_sum(array, perc=(0,99)):
  '''
  loops through each band in the array, 
  computes sobel edge value, and returns the 
  SUM total edge value from all bands in all_edges array
  '''
  bands = np.arange(array.shape[0])
  # set a shape for the edge output to match the input
  shape = array[0].shape
  all_edges = np.zeros(shape, dtype=np.float)
  for band in bands:
    print("edge sum band number:", band)
    # rescale band
    p_low, p_high = np.percentile(array[band], perc)
    edge_band = exposure.rescale_intensity(
            array[band],
            in_range=(p_low, p_high),
            out_range=(0, 255))

    # find edges for each band 
    edges = filters.sobel(edge_band)
    edges = exposure.rescale_intensity(edges, out_range=(0, 255))
    all_edges += edges
    
  all_edges = filters.gaussian(all_edges, 1)
  return all_edges  

def canny_edge(input_array, sigma = 3):
    canny_edges = feature.canny(input_array, sigma = sigma)
    return canny_edges

def combine_edges(edges_max, edges_sum, max_weight=1, sum_weight=1):
    return (max_weight*edges_max) + (sum_weight*edges_sum)