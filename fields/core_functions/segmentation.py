# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 13:07:28 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Segmentation functions
import skimage
from skimage import feature, measure, morphology
import scipy.ndimage as ndi

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

# watershed segmentation
def watershed_segments(input_dt, input_markers):
    print("Watershed segmentation")
    labels = morphology.watershed(-input_dt, input_markers)
    
    return labels

# plot input image with edges
def plot_labels_on_input_img(input_img, labels):        
    fig = plt.figure(figsize=(12,12))
    plt.imshow(mark_boundaries(input_img, labels), cmap='PiYG')

















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