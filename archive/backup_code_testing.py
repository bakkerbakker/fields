# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:34:42 2019

BACKUP

@author: jesse
"""


### Code Testing Document
import os
# for fields library to work, cwd needs to be: C:\Users\jesse\Documents\grad school\masters research\code
print(os.getcwd())

#%%
import imp
import fields
from fields import *
#from fields import utilities, edges, mask, ndvi, segmentation, read_rasters, write_shapefiles
from matplotlib import pyplot as plt
import scipy.ndimage as ndi
import numpy as np
#%%
imp.reload(fields)
#%%

## Create NDVI stack and Edge Map from timestack of rasters
#rasters_large = ["Sentinel_Stack_20170513_BGRN_TargetAreaClip.tif",
#             "Sentinel_Stack_20170622_BGRN_TargetAreaClip.tif", 
#             "Sentinel_Stack_20170720_BGRN_TargetAreaClip.tif", 
#             "Sentinel_Stack_20170829_BGRN_TargetAreaClip.tif",      
#             "Sentinel_Stack_20171020_BGRN_TargetAreaClip.tif"]

# rasters clipped to smaller study area for quicker processing/testing
rasters = ["Sentinel_Stack_20170513_BGRN_TargetArea2_Clip.tif",
             "Sentinel_Stack_20170622_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170720_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170801_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170829_BGRN_TargetArea2_Clip.tif",      
             "Sentinel_Stack_20171020_BGRN_TargetArea2_Clip.tif"]

folder_path = "C:/Users/jesse/Documents/GISData/Sentinel2/"
print(folder_path + rasters[0])
#%%

### Is there a way to add this to the module? 
### I get errors saying that the embedded fields functions aren't defined.

### TO DO: get this into the module.
### ***Should it be split up to return each part separately?***
### 1. Read raster list, for each raster calculate NDVI
### 2. Return full raster stack and NDVI stack
### 3. Pass full raster stack to Edge detection functions
### $. Result in full imagery stack, NDVI stack, two cumulative edge arrays

### this version adds NDVI to the band stack prior to the edge detection
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
        ndvi = None
        edges_max = None
        edges_sum = None
        count += 1
        
    return ndvi_stack, cumulative_edges_max, cumulative_edges_sum

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
        ndvi = None
        edges_max = None
        edges_sum = None
        count += 1
        
    return ndvi_stack, cumulative_edges_max, cumulative_edges_sum
#%%

ndvi_stack, edges_max, edges_sum = time_stack_ndvi_and_edges(folder_path, rasters)

#%%
# visualize the edges and NDVI range

visualize_2D_array_0_1000(edges_max, title="Edges Max")
visualize_2D_array_0_1000(edges_sum, title="Edges Sum")
visualize_2D_array_0_1000(ndvi_range_from_stack(ndvi_stack), title="NDVI Range")
#%%
# visualize NDVI min and max

visualize_2D_array_0_1000(ndvi_max_from_stack(ndvi_stack), title="NDVI Max")
visualize_2D_array_0_1000(ndvi_min_from_stack(ndvi_stack), title="NDVI Min")
#%%
def visualize_2D_array_0_1000(array, title = '', cmap = 'viridis'):
    fig = plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.axis('off')
    plt.imshow(array[0:1000, 0:1000], cmap = cmap)

# Closer look at edge max
fig = plt.figure(figsize=(10, 10))
plt.title("")
plt.axis('off')
plt.imshow(cumulative_edges_max[0:1000, 0:1000], cmap = "viridis")

# Closer look at edge sum
fig = plt.figure(figsize=(10, 10))
plt.title("")
plt.axis('off')
plt.imshow(cumulative_edges_sum[0:1000, 0:1000], cmap = "viridis")

cumulative_edges_combined = normalize((cumulative_edges_max) + (2*cumulative_edges_sum))

# Closer look at the combined edges
fig = plt.figure(figsize=(10, 10))
plt.title("")
plt.axis('off')
plt.imshow(cumulative_edges_combined[0:1000, 0:1000], cmap = "viridis")

#%%

# Closer look at NDVI range
print(ndvi_stack.shape)
ndvi_range = np.amax(ndvi_stack, axis = 2) - np.amin(ndvi_stack, axis = 2)
ndvi_range = 1 - normalize(ndvi_range)
# plot ndvi range
fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(ndvi_range[0:1000, 0:1000], cmap = "RdYlGn")

#print(np.amax(ndvi_range), np.amin(ndvi_range))

#def ndvi_range(ndvi_stack):
#   return np.amax(ndvi_stack, axis = 2) - np.amin(ndvi_stack, axis = 2)

#%%

# Create Mask from NDVI and Edges
mask_combo = normalize(3*cumulative_edges_combined + ndvi_range)
# plot mask combo
fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(mask_combo[0:1000, 0:1000], cmap = "viridis")

#mask_combo_edges = filters.sobel(mask_combo)
#mask_combo_edges = exposure.rescale_intensity(mask_combo_edges, out_range=(0, 255))
#mask_combo_edges = 1 - normalize(mask_combo_edges)

#fig = plt.figure(figsize=(10, 10))
#plt.axis('off')
#plt.imshow(mask_combo_edges[1000:2000, 1000:2000], cmap = "viridis")

#%%

fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(return_masked_array(mask_combo[0:1000, 0:1000], 75, 100))

#%%

# Convert mask to binary
binary_mask = np.zeros_like(mask_combo)
binary_mask[normalize(mask_combo) < .15] = 1
binary_mask = ndi.binary_fill_holes(binary_mask)

fig = plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(binary_mask[0:1000, 0:1000])
#%%

# check histogram of edges
from skimage.exposure import histogram
hist, hist_centers = histogram(mask_combo[0:1000, 0:1000])

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(mask_combo[0:1000, 0:1000], cmap=plt.cm.gray, interpolation='nearest')
axes[0].axis('off')
axes[1].plot(hist_centers, hist, lw=2)
axes[1].set_title('histogram of gray values')

#%%

# Segment the imagery
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

#%%

# skimage threshold testing on the mask_combo


#%% 

### Watershed segmentation

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
