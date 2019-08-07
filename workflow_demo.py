# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:24:43 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Workflow Demo ###

#%%

### Import packages
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


#%%
### Read in rasters to create mask

# read list of rasters to derive ndvi_stack (ndvi array for each time slice) and cumulative edges

mask_inputs = {'folder_path' : folder_path,
               'rasters' : rasters,
               'max_edge_weight' : 1,
               'sum_edge_weight' : 4,
               'ndvi_weight' : 1,
               'edges_weight' : 2,
               'binary_threshold' : .16}

mask_array = mask_func(**mask_inputs)
#%%
### Create rgb image for fz segmentation and RAG Merge functions

rgb_img_inputs = {'folder_path' : folder_path,
                  'rasters' : rasters,
                  'raster_image_index' : 4,
                  'gamma': .5, 
                  'clip_val_r': 1500, 
                  'clip_val_g': 1500, 
                  'clip_val_b': 1500}

rgb_image = rgb_img_func(**rgb_img_inputs)

#%%
### Segment imagery with mask applied

# RGB Image to felzenswsalb segmentation
segmentation_fz_inputs = {'fz_scale': 50,
                          'fz_sigma': 0.3,
                          'fz_min_size': 50}

labels_fz = segmentation_fz_func(rgb_image = rgb_image,
                                 mask_array = mask_array,
                                 **segmentation_fz_inputs)

#%%
# watershed segmentation
segmentation_ws_inputs = {'ws_min_dist':3,
                          'ws_compactness': 0}

labels_ws = segmentation_ws_func(mask_array = mask_array,
                                 edges_array = mask_array,
                                 **segmentation_ws_inputs)

#%%

### Merge segments
merge_inputs = {'merge_hierarchical_threshold': 0.07}

merged_labels_fz = merge_segments_func(input_img = rgb_image, 
                                    input_labels = labels_fz, 
                                    **merge_inputs)

merged_labels_ws = merge_segments_func(input_img = rgb_image, 
                                    input_labels = labels_ws, 
                                    **merge_inputs)

#%%

# plot merged labels over rgb image
fig = plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(mark_boundaries(rgb_image[0:1000, 0:1000], labels_fz[0:1000, 0:1000]))

#%%

# plot merged labels over rgb image
fig = plt.figure(figsize=(10,10))
plt.axis('off')
plt.imshow(mark_boundaries(rgb_image[0:1000, 0:1000], merged_labels_ws[0:1000, 0:1000]))
#%%

### Need to update this so that null/zero value polygons (non-crop areas) don't get included in the shp

### Save to shapefile
write_shp_inputs = {'raster_folder_path' : folder_path,
                    'rasters' : rasters,
                    'raster_image_index' : 4,
                    'output_folder_path': 'outputs/',
                    'output_file': 'ws_test.shp'}    
    
write_shapefile_func(merged_labels = merged_labels, 
                     **write_shp_inputs)

#%%
    
### Compare to validation fields

validation_inputs = {'ref_fp':"data/reference_fields/DigitizedRefFields_CDLsummaryjoin_TargetArea2.shp",
                     'bbox_fp': "data/reference_fields/TargetArea2.shp",
                     'test_fp':"outputs/test.shp"}

overlay_test_df = validation_fields_comparison_func(**validation_inputs)

