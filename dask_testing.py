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
import dask
#%%
# Run to reload the fields package after updating code in the individual .py files
imp.reload(fields)
print(dir(fields))

#%%

