# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:07:28 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Segmentation functions
import skimage
from skimage import feature, measure, morphology, segmentation, exposure
import scipy.ndimage as ndi
import numpy as np
from fields.IO.read_rasters import read_tif_to_array, prep_rgb_image
from fields.core_functions.mask import apply_mask

# canny edge detection
def array_2D_canny_edges(input_img, canny_sigma = 1):
    print("Canny edge detection")
    edges = skimage.feature.canny(input_img, canny_sigma)

    return edges

# distance from edges
def distance_from_edges(input_edges):
    # from scipy.ndimage import distance_transform_edt
    print("Computing distance from edges")
#    dt = ndi.distance_transform_edt(~input_edges)
    dt = ndi.distance_transform_edt(input_edges)    
    return dt
    
# local max - Need to understand this step better
def local_max(input_dt, min_dist = 5):
    # compute local max to get "fountains" for watershed segmentation
    print("Local Max for Watershed Pooling")
    local_max = feature.peak_local_max(input_dt, indices=False, 
                                       min_distance=min_dist)
    # plot local max
#    peak_idx = feature.peak_local_max(input_dt, indices=True, 
#                                      min_distance=min_dist)
#    fig = plt.figure(figsize=(12,12))
#    plt.imshow(peak_idx)
    
    # label markers
    markers = measure.label(local_max)
    
    return markers

# watershed segmentation with mask
def watershed_segments(input_dt, input_markers, mask = None, compactness = 0):
    print("Watershed segmentation")

    labels = morphology.watershed(-input_dt, input_markers, mask = mask, compactness = compactness)

    return labels

# plot input image with edges
def plot_labels_on_input_img(input_img, labels):        
    fig = plt.figure(figsize=(12,12))
    plt.imshow(mark_boundaries(input_img, labels), cmap='PiYG')




#### Combined functions for workflow
def segmentation_fz_func(rgb_image, mask_array, **segmentation_fz_inputs):
    # define inputs
    mask = mask_array
    fz_scale = segmentation_fz_inputs['fz_scale']
    fz_sigma = segmentation_fz_inputs['fz_sigma']
    fz_min_size = segmentation_fz_inputs['fz_min_size']
                               

    # Apply mask to rgb image
    image_to_segment = apply_mask(rgb_image, 
                                  np.logical_not(mask), 
                                  masked_value=0)
    # Segment the image
    labels = segmentation.felzenszwalb(image_to_segment, 
                          scale = fz_scale, 
                          sigma=fz_sigma, 
                          min_size=fz_min_size)
    # return labeled segments
    return labels


def segmentation_ws_func(mask_array, edges_array, **segmentation_ws_inputs):
    """
    During testing, the edges_array was simply the binary mask array,
    but this can be tweaked. For instance, another option was a canny
    edge output of the combined edges array
    """
    # define inputs within function
    mask = mask_array
    edges = edges_array
    ws_min_dist = segmentation_ws_inputs['ws_min_dist']
    ws_compactness = segmentation_ws_inputs['ws_compactness']
    
    # map distance from mask edges
    # for now, we will just pass the mask array
    dt = distance_from_edges(edges)
    # create markers for watershed pooling at local max distance
    markers = local_max(dt, min_dist=ws_min_dist)
    # watershed segmentation
    labels = watershed_segments(input_dt = dt, 
                                input_markers = markers, 
                                mask = mask, 
                                compactness = ws_compactness)
    # return labeled segments
    return labels












#### NOT WORKING YET, NEED TO TEST/DEBUG
#def watershed_segment(edges, footprint_size):
#    '''
#    create markers in low info regions of edges, and use watershed to
#    produce segments.
#    '''
#    print('beginning segmentation')
#    edges_i = 1 - edges  # invert edges for peak_local_max
#    markers = feature.peak_local_max(
#            edges_i,
#            footprint=morphology.disk(footprint_size),
#            indices=False)
#    markers, _ = ndi.label(markers)
#    segments = morphology.watershed(edges, markers)
#    return segments

#def watershed_segment_mask(edges, footprint_size, input_mask):
#    '''
#    create markers in low info regions of edges, and use watershed to
#    produce segments.
#    '''
#    click.echo('beginning segmentation')
#    edges_i = 255 - edges  # invert edges for peak_local_max
#    markers = feature.peak_local_max(
#            edges_i,
#            footprint=morphology.disk(footprint_size),
#            indices=False)
#    markers, _ = ndi.label(markers)
#    segments = morphology.watershed(edges, markers, mask=input_mask)
#    return segments