# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 15:09:20 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### dask testing document

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
import dask.array as da
from scipy.ndimage.filters import gaussian_filter
#%%
# Run to reload the fields package after updating code in the individual .py files
imp.reload(fields)
print(dir(fields))

#%%

"""
Anticipated issues:
    - how to handle inputs for the workflow function? can the dictionaries for each step be stored inside the function?
    - saving the output to shapefile, will need to stitch the tiles back together or else all of them will overlap spatially
    - how to handle feature edges at tile boundaries?
"""

#%%



### get shape of the raster
# raster folder
folder_path = "data/rasters/"
# rasters clipped to smaller study area for quicker processing/testing
rasters = ["Sentinel_Stack_20170513_BGRN_TargetArea2_Clip.tif",
             "Sentinel_Stack_20170622_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170720_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170801_BGRN_TargetArea2_Clip.tif", 
             "Sentinel_Stack_20170829_BGRN_TargetArea2_Clip.tif",      
             "Sentinel_Stack_20171020_BGRN_TargetArea2_Clip.tif"]
# raster filepath
raster_fp = folder_path + rasters[4]
with rasterio.open(raster_fp) as src:
    width = src.meta['width']
    height = src.meta['height']

print(width, height)
#%%
### make the list of chunks, an index of how the raster will be tiled
chunk_size = 1000
chunk_list = []

for r in range(0, height, chunk_size):
    for c in range(0, width, chunk_size):
        chunk_list.append([r,c,chunk_size,chunk_size])

print(chunk_list)
#%%
### figure out how the map_blocks() function works
# get the raster array into a dask array
np_array = fields.read_tif_to_array(raster_fp)
np_array = np_array[:,:1000,:1000]
np_array.shape
#%%
#np_array = None
g = da.from_array(np_array, chunks = (4, 100, 100))
g.chunks
#%%
print(g.blocks)


#%%
### make the workflow function to pass to the dask.map_blocks() function
def test_func_ndvi(block):
    if block_id not equal "__dummy_id__"
    
    ## calculate ndvi
    ndvi = fields.compute_ndvi(block[3,:,:],block[2,:,:])
    ## check values
    return ndvi

def test_func_edgesum(block):
    ## calculate edges
    edges_sum = fields.edges_from_3Darray_sum(block)
    ## check values
    return edges_sum    

def test_func_print(block, block_id = None):
    if block_id == (0,9,9):
        print("block:", block, block_id)
    return block
    
def func(block):
    return gaussian_filter(block, sigma=1)

#ndvi_test = g.map_blocks(test_func_print)
test_out = g.map_blocks(test_func_ndvi, chunks=(4, 100, 100))
#%%

test_out.compute()
#da.overlap.trim_internal(test_out, {0,1,1})
#%%
### pass the chunk_list 









    