# -*- coding: utf-8 -*-
"""
Created on Thu May 30 12:16:54 2019

@author: jesse bakker (bakke557@umn.edu)
"""

### Write to shapefile

# Improt libraries
import fiona
import rasterio
from rasterio import features

            
def read_raster_to_transform(filepath):
    ''' Takes an input raster and returns the "transform" metadata with
        spatial transformation data as a dictionary. This is used for 
        reprojecting the vectorzed segmentation as a shapefile.'''
        
    with rasterio.open(filepath) as src:
        transform = src.transform
        
    return transform
        
def read_raster_to_crs(filepath):
    ''' Takes an input raster and returns the coordinate reference system
        "crs" as a string, ex: 'EPSG:32614'. Used for reprojecting the vectorzed 
        segmentation as a shapefile in the "write_segments_to_shapefile" function.'''
 
    with rasterio.open(filepath) as src:
        profile = src.profile
        crs = str(profile['crs'])
   
    return crs

def write_segments_to_shapefile(input_array, src_transform, src_crs, output_file):
    '''
    This function takes an array (meant for a raster that has already been segmented)
    and writes the polygonized raster to a shapefile.

    input_array: raster to by polygonized and exported
    src_transform: the "transform" spatial metadata from the rasterio.read() of the source raster
    src_crs: the coordinate reference system from the source raster, as a string. ex: 'EPSG:32614'
    output_file: file path/name ending with '.shp' for the output
    '''
    # set input array to integer
    array_int = input_array.astype(int)
    
    # polygonize input raster to GeoJSON-like dictionary with rasterio.features.shapes
    # src_transform comes from the source raster that holds the spatial metadata
    results = ({'geometry': s, 'properties': {'raster_val': v}}
          for i, (s, v) in enumerate(features.shapes(array_int, mask=None, transform = src_transform)))
    geoms = list(results)
    
    # establish schema to write into shapefile
    schema_template = {
    'geometry':'Polygon', 
    'properties': {
        'raster_val':'int'}}
    
    # src_crs is the coordinate reference system from the source raster that holds the spatial metadata
    with fiona.open(output_file, 'w', driver='Shapefile', schema = schema_template, crs = src_crs) as layer:
        # loop through the list of raster polygons and write to the shapefile
        for geom in geoms:
            layer.write(geom)

#### Combined function for workflow
def write_shapefile_func(merged_labels, **write_shp_inputs):
    # Set output location
    output_file = write_shp_inputs['output_folder_path'] + write_shp_inputs['output_file']
    
    # Get geometadata from input raster, this should be one of the rasters used 
    # in the first step of the process to ensure that the shp output is correct.
    raster_folder_path = write_shp_inputs['raster_folder_path']
    rasters = write_shp_inputs['rasters']
    raster_image_index = write_shp_inputs['raster_image_index']
    raster_filepath = raster_folder_path + rasters[raster_image_index]
    
    transform = read_raster_to_transform(raster_filepath)
    crs = read_raster_to_crs(raster_filepath)
    
    # write labeled/merged segments from the "merge_segments" function to a shapefile
    write_segments_to_shapefile(input_array = merged_labels,
                                src_transform = transform,
                                src_crs = crs,
                                output_file = output_file)