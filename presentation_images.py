# -*- coding: utf-8 -*-


# set working directory for running the fields module
import os
# for fields library to work, cwd needs to be: 
# C:\Users\jesse\Documents\grad school\masters research\code\fields_library
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
import dask.array as da
import geopandas as gpd

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
             "Sentinel_Stack_20170829_BGRN_TargetArea2_Clip.tif",      
             "Sentinel_Stack_20171020_BGRN_TargetArea2_Clip.tif"]

print(folder_path + rasters[0])

#%%
### time series of RGB, NDVI, edges

date_list = ['5/13', '6/22','7/20','8/29','10/20']
image_types = ["RGB Composite", "NDVI", "Edges Max", "Edges Sum"]
fig, axes = plt.subplots(figsize=(16, 12), ncols=5, nrows=4, sharey=True, sharex=True)

for raster, col, date in zip(rasters, axes.T, date_list):
    # define raster path
    filepath = folder_path + raster
    # read raster to array
    band_stack = read_tif_to_array(filepath)
    # rgb image
    rgb_image = prep_rgb_image(band_stack, 
                               gamma=.5, 
                               clip_val_r = 1500, 
                               clip_val_g = 1500, 
                               clip_val_b = 1500)
    # NDVI
    ndvi = compute_ndvi(band_stack[3], band_stack[2])
    # calculate edges from band stack
    edges_max = edges_from_3Darray_max(band_stack)
    edges_sum = edges_from_3Darray_sum(band_stack)
    
    col[0].set_title(date, size = 18)
    col[0].imshow(rgb_image[0:300,0:300])
    col[1].imshow(ndvi[0:300,0:300], cmap="PiYG")
    col[2].imshow(edges_max[0:300,0:300], cmap="viridis")
    col[3].imshow(edges_sum[0:300,0:300], cmap="viridis")
#    col[0].spines.set_visible(False)
#    col[1].spines.set_visible(False)
#    col[2].spines.set_visible(False)
#    col[3].spines.set_visible(False)

for first_in_row, i in zip(axes[:,0], image_types):
    first_in_row.set_ylabel(i, size=18)

plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

#%%

