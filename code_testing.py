# -*- coding: utf-8 -*-
"""
Created on Thu May 30 13:58:53 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Code Testing Document

"""
TO DO:
    - finish the segmentation step
    - create the merge step, which will require some exploration around what data 
        to use for the merge and how best to cluster the segments into similar, contiguous areas
    - implement the write_shapefile code: already written and working, just need to integrate it
        into the workflow
    - accuracy assessment, need to establish metrics for: 
        - field count/total crop area
        - over-segmentation and undersegmentation
        - boundary accuracy
    - set up a testing framework. **ERIC, I'll need some guidance on this step**
"""

#%%
# set working directory for running the fields module
import os
# for fields library to work, cwd needs to be: C:\Users\jesse\Documents\grad school\masters research\code\fields_library
# Set file path below to point to the folder that contains the fields module
code_fp = 'C://Users//jesse//Documents//grad school//masters research//code//fields_library'
os.chdir(code_fp)
print(os.getcwd())

#%%
import imp
import fields
from fields import *
#from fields import utilities, edges, mask, ndvi, segmentation, read_rasters, write_shapefiles
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import numpy as np

import dask

#%%
# Run to reload the fields package after updating code in the individual .py files
imp.reload(fields)
print(dir(fields))

#%%
# folder path to directory containing raster data
# original folder_path = "C:/Users/jesse/Documents/GISData/Sentinel2/"
folder_path = "data/rasters/"

# rasters clipped to smaller study area for quicker processing/testing
rasters = ["Sentinel_Stack_20170513_BGRN_TargetArea2_Clip.tif",
             "Sentinel_Stack_20170622_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170720_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170801_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170829_BGRN_TargetArea2_Clip.tif",      
             "Sentinel_Stack_20171020_BGRN_TargetArea2_Clip.tif"]

print(folder_path + rasters[0])
#%%

### SOLVED: IMPORT THE SPECIFIC FUNCTIONS TO THE INDIVIDUAL .PY FILES
### Is there a way to add this to the module? 
### I get errors saying that the embedded fields functions aren't defined.

### ***STRUCTURAL QUESTION: Should it be split up to return each part separately?***
### ***What is the best way to do IO for separate tiles and separate imagery dates???***
### re: time_stack_ndvi_and_edge() function in mask.py
### 1. Read raster list, for each raster calculate NDVI
### 2. Return full raster stack and NDVI stack
### 3. Pass full raster stack to Edge detection functions
### 4. Result in full imagery stack, NDVI stack, two cumulative edge arrays
### 5. What other derived layers do we need to produce from the full stack? Min/Max/Range etc.

#%%
# read list of rasters to derive ndvi_stack (ndvi array for each time slice) and cumulative edges
ndvi_stack, edges_max, edges_sum = time_stack_ndvi_and_edges(folder_path, rasters)


#%%

### IN DEVELOPMENT

### QUESTION FOR ERIC: How to make the function within the for loop modular? 
### It creates the cumulative edges layer and the NDVI stack in the first loop
### but needs to be able to reference them in the subsequent loops.
### However, in the second loops they are undefined local variables instead of 
### persisting from the first loop.

### Set up the for loop function that opens the raster stack to take 
### a function as an argument to process within the loop
def time_stack_for_loop(folder_path, rasters, loop_function, return_vals):
    '''
    function to read in a list of raster files and loop through each raster.
    the function within the loop ("loop_function") needs to be defined externally
    and should take the "band_stack" as an input parameter.
    the values that will be returned, passed as a list to the function.
    '''
    
    count = 0
    
    for raster in rasters:
        # define raster path
        filepath = folder_path + raster
        
        # read raster to array
        band_stack = read_tif_to_array(filepath)
        
        # pass band_stack to loop function
        return_vals = loop_function(band_stack, count)
        
        # update count for each loop
        count += 1

    return return_vals


def ndvi_and_edges_for_loop(band_stack, count):
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
    
    return ndvi_stack, cumulative_edges_max, cumulative_edges_sum
    
return_vals = ["ndvi_stack", "cumulative_edges_max", "cumulative_edges_sum"]
#return_vals_test = ["test"]
ndvi_stack, edges_max, edges_sum = time_stack_for_loop(folder_path, rasters, ndvi_and_edges_for_loop, return_vals_test)


#%%
# check ndvi_stack output for expected shape/results
print(ndvi_stack[:,:,1].shape)
visualize_2D_array_0_1000(ndvi_stack[:,:,2], title="NDVI Band Example")

#%%
# visualize the edges and NDVI range
visualize_2D_array_0_1000(edges_max, title="Edges Max")
visualize_2D_array_0_1000(edges_sum, title="Edges Sum")
visualize_2D_array_0_1000(ndvi_range_from_stack(ndvi_stack), title="NDVI Range")
visualize_2D_array_0_1000(ndvi_max_from_stack(ndvi_stack), title="NDVI Max")
visualize_2D_array_0_1000(ndvi_min_from_stack(ndvi_stack), title="NDVI Min")
#%%
# create combined edges layer, with variable weights for sum and max layers
# max identifies more boundaries that only appear in one or two time slices, more noisy
# sum identifies persistent edges, less noisy but some edges are faint or not identified
edges_combined = normalize(combine_edges(normalize(edges_max), normalize(edges_sum), 
                                         max_weight=1, sum_weight=4))

# Closer look at the combined edges
visualize_2D_array_0_1000(edges_combined, title="Edges Combined")

#%%

# Closer look at NDVI range, normalized
visualize_2D_array_0_1000(1-normalize(ndvi_range_from_stack(ndvi_stack)), 
                          title="Normalized Inverted NDVI Range",
                          cmap = "RdYlGn")

#%%

# Create Mask from NDVI and Edges
mask_combo = create_combined_mask(ndvi = 1 - normalize(ndvi_range_from_stack(ndvi_stack)), edges = edges_combined,
                     ndvi_weight = 1, edges_weight = 2)

visualize_2D_array_0_1000(mask_combo, title="Combined Mask Array, NDVI Range and Cumulative Edges")


#%%
from skimage.segmentation import (morphological_chan_vese, 
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

# testing Morphological GAC

def store_evolution_in(lst):
    """
    returns a callback function to store the evolution of the
    level sets in the given list.
    """
    
    def _store(x):
        lst.append(np.copy(x))
        
    return _store

image = mask_combo
gimage = inverse_gaussian_gradient(image)

# initial level set
init_ls = np.zeros(image.shape, dtype=np.int8)
init_ls[10:-10, 10:-10] = 1
# list with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_geodesic_active_contour(gimage, 20, init_ls, 
                                           smoothing=1, balloon = -1,
                                           threshold = 0.25,
                                           iter_callback = callback)


#%%

### Display morph snakes
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
ax = axes.flatten()

ax[0].imshow(image)
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors = 'r')
ax[0].set_title("Morphologiacal GAC segmentation", fontsize=12)

ax[1].imshow(ls, cmap='gray')
ax[1].set_axis_off()
contour = ax[1].contour(evolution[0], [0.5], colors='g')
contour.collections[0].set_label("Iteration 0")
contour = ax[1].contour(evolution[10], [0.5], colors='y')
contour.collections[0].set_label("Ieration 10")
contour = ax[1].contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("iteration 20")
ax[1].legend(loc='upper right')
title = "Morphological GAC Evolution"
ax[1].set_title(title, fontsize=12)

fig.tight_layout()
plt.show()
#%%

# testing what it looks like to pass the combined edges layer through an edge filter (edge of the edges)
# not used later in the workflow here but this is potentially useful for offsetting fields from noisy boundary pixels
mask_combo_edges = filters.sobel(mask_combo)
mask_combo_edges = normalize(mask_combo_edges)

#visualize_2D_array_0_1000(mask_combo_edges)
# pass this to the watershed segmentation to compute distance map
mask_combo_canny_edges = array_2D_canny_edges(mask_combo, 2.8)
visualize_2D_array_0_1000(mask_combo_canny_edges)

#%%

# Masked array testing
lower_mask_pct = 0
upper_mask_pct = 75
visualize_2D_array_0_1000(return_masked_array(mask_combo, lower_mask_pct, upper_mask_pct), 
                          title="Masked Array by Percent of Pixels: Lower Perc. = " + str(lower_mask_pct) + ", Upper Perc = " + str(upper_mask_pct))

#%%
# Convert mask to binary array, 
binary_threshold = .16
binary_mask = create_binary_mask(mask_combo, threshold=binary_threshold, fill_holes=True)
visualize_2D_array_0_1000(binary_mask, title="Binary Mask, Threshold:" + str(binary_threshold))

#%%
# Create ternary (three-category) mask, identifying Field, Not-field, Uncertain

lower_thresh = .15
upper_thresh = .25
ternary_mask = create_ternary_mask(mask_combo, lower_thresh = lower_thresh, upper_thresh = upper_thresh)
visualize_2D_array_0_1000(ternary_mask, 
                          title="Ternary Mask, Threshold: Lower Thresh = " + str(lower_thresh) + ", Upper Thresh = " + str(upper_thresh),
                          cmap='viridis')


#%%

### This is pulled from other code, probably not useful in this workflow but could be a good reference for combining masks
def prep_mask(ndvi_array, ndvi_maskpct_low, edge_array, edge_maskpct_high):
  ndvi_mask = returnMaskedArray(ndvi_array, ndvi_maskpct_low, 100).astype(bool)
  edge_mask = returnMaskedArray(edge_array, 0, edge_maskpct_high).astype(bool)
  
  mask_array = (ndvi_mask == 1) & (edge_mask == 1)
  return mask_array

#%%

# read RGB from tif raster
raster_image_index = 4
raster_fp = folder_path + rasters[raster_image_index]
print(raster_fp)
rgb_image = prep_rgb_image(read_tif_to_array(raster_fp), gamma=.5, clip_val_r = 1500, clip_val_g = 1500, clip_val_b = 1500)

visualize_3_band_image_array_0_1000(rgb_image, title="RGB image test, raster image index:" + str(raster_image_index))

#%%

# Apply mask to rgb image
image_to_segment = apply_mask(rgb_image, np.logical_not(binary_mask), masked_value=0)
visualize_3_band_image_array_0_1000(image_to_segment)

#%%

## Active contour segmentation experimentation
from skimage.segmentation import random_walker
markers = ternary_mask
labels_rw = random_walker(mask_combo, markers, beta = 100, mode='cg_mg')

fig = plt.figure(figsize=(10,10))
plt.imshow(labels_rw[0:1000, 0:1000])

#%%

# segment rgb image
from skimage.segmentation import felzenszwalb, mark_boundaries
labels = felzenszwalb(image_to_segment, scale = 50, sigma=.3, min_size=50)

fig = plt.figure(figsize=(10,10))
plt.imshow(mark_boundaries(image_to_segment[0:1000, 0:1000], labels[0:1000, 0:1000]))


### Try morphological snakes: https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_morphsnakes.html#sphx-glr-auto-examples-segmentation-plot-morphsnakes-py


#%%

### ERIC: the rest of the code below is still in progress ###
# Segment the imagery   
### Sample workflow
#canny_edges = array_2D_canny_edges(binary_mask, 1)

## pass this to the watershed segmentation to compute distance map
#mask_combo_canny_edges = array_2D_canny_edges(mask_combo, 2.5)
#dt = distance_from_edges(mask_combo_canny_edges)

dt = distance_from_edges(binary_mask)
markers = local_max(dt, min_dist=3)
labels = watershed_segments(dt, markers, mask = binary_mask)
fig = plt.figure(figsize=(10,10))
plt.imshow(mark_boundaries(image_to_segment[0:1000, 0:1000], labels[0:1000, 0:1000]))
#watershed_labels = watershed_segments(input_maskedArray,distanceFromEdges(array2D_toCannyEdges(input_maskedArray, **segmentation_config)),
                                         #localMax(distanceFromEdges(array2D_toCannyEdges(input_maskedArray, **segmentation_config)),**segmentation_config))

#%%
# segmentation region properties - 
# cluster segments based on region properties
 # merge segments

### TEST: region props table https://scikit-image.org/docs/dev/api/skimage.measure.html#regionprops-table                                        
### Process idea: loop through different bands/intermediate data layers as inputs for the "intensity image"
### to be passed to the skimage.measure.regionprops_table() function and append each as a column in a pandas df
### where each row is a segmented region. Then use that table for clustering. 
### Question for Eric: Is this the right data structure for this?                                         
                                         
# opportunity here for pulling properties from other arrays i think
# this might be better suited to take the property type as an input argument
# see regionprops: http://scikit-image.org/docs/dev/api/skimage.measure.html?#skimage.measure.regionprops
# options: area, bounding box area, eccentricity, perimeter, 
def measureRegionProps_toRegions(input_labels, property_img):
    regions = measure.regionprops(input_labels, intensity_image=property_img)
    
    return regions

def getRegionMeanIntensity_toRegionMeans(input_regions):
    
    region_means = [r.mean_intensity for r in input_regions]
    print("Check intensity histogram for identifying KMeans clusters")
    plt.hist(region_means, bins=50)
    plt.show()
        
    return region_means

def getRegionEccentricity_toRegionMeans(input_regions):
    
    region_means = [r.eccentricity for r in input_regions]
    print("Check intensity histogram for identifying KMeans clusters")
    plt.hist(region_means, bins=50)
    plt.show()
    
    return region_means

# KMeans clustering of segments
def clusterKMeans_toClassifiedLabels(input_labels, input_regions, input_region_means, n_clusters = 2):
    # clustering model
    model = KMeans(n_clusters=n_clusters)
    
    regions = input_regions

    region_means = np.array(input_region_means).reshape(-1,1)
    model.fit(np.array(region_means).reshape(-1,1))
    print("Model CLuster Centers:", model.cluster_centers_)
    area_labels = model.predict(region_means)
    
    print("Classify objects")
    classified_labels = input_labels.copy()
    for area, region in zip(area_labels, regions):
        classified_labels[tuple(region.coords.T)] = area
    
    return classified_labels


# plot classified labels
def plotClassifiedKMeansLabels(input_img, classified_labels):

    # plot input image with edges
    fig = plt.figure(figsize=(12,12))
    plt.imshow(color.label2rgb(classified_labels, image=input_img))







#%% 

### Watershed segmentation
from skimage import morphology
from scipy import ndimage as ndi

#mask_combo = normalize(mask_combo)

markers = np.zeros_like(mask_combo)
markers[mask_combo < .12] = 1
#markers[mask_combo > .12] = 0

fill_markers = ndi.binary_fill_holes(markers)
new_edges = 0.5*fill_markers + mask_combo

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(new_edges[0:1000, 0:1000],  cmap = "viridis", interpolation='nearest')
ax.set_title('markers')
ax.axis('off')

#segmentation = morphology.watershed(mask_combo, markers)
#fig, ax = plt.subplots(figsize=(8, 8))
#ax.imshow(segmentation[3000:4000, 2000:3000], cmap=plt.cm.nipy_spectral, interpolation='nearest')
#ax.set_title('segmentation')
#ax.axis('off')

#from rasterio.plot import show_hist
#import skimage

#show_hist(mask_combo, bins = 50)

#canny_edges = fields.edges.canny_edge(mask_combo, sigma = 1)
#fig, ax = plt.subplots(figsize=(8, 8))
#ax.imshow(canny_edges[0:1000, 0:1000],  cmap = "viridis", interpolation='nearest')
#ax.set_title('markers')
#ax.axis('off')

watershed = morphology.watershed(mask_combo)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(watershed,  cmap = "viridis", interpolation='nearest')
ax.set_title('markers')
ax.axis('off')
