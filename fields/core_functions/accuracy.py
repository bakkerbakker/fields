# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:53:06 2019

@author: jesse bakker (bakke557@umn.edu)
"""

"""
Accuracy assessment, summary statistics, and validation for field outputs
"""

import geopandas as gpd
import pandas as pd

# read a shapefile filepath string to a geodataframe
def shapefile_to_gpd_df(shp_fp, bbox=None):
    """
    read a shapefile filepath string to a geodataframe
    """
    gpd_df = gpd.read_file(shp_fp, bbox = bbox)
    return gpd_df

# convert both shp to same crs, in this case 'epsg:32614' WGS 84 / UTM zone 14N
def convert_crs(input_gpd_df, crs_dest = {'init': 'epsg:32614'}):
    """
    Change the coordinate reference system of a GeoDataFrame.
    By default it will change it to 'epsg:32614' WGS 84 / UTM zone 14N
    in order to calculate area accurately for regions in Minnesota.
    """
    if input_gpd_df.crs == crs_dest:
        output_gpd_df = input_gpd_df.copy()
    else:
        output_gpd_df = input_gpd_df.to_crs(crs_dest)        
    return output_gpd_df

# get bounding box of study area
def get_bbox(bbox_df):
    """
    Return a list of bbox coordinates in the order: ['minx','miny','maxx','maxy']
    """
    bounds_list = ['minx','miny','maxx','maxy']
    bbox = []
    for bound in bounds_list:
        bbox.append(convert_crs(bbox_df).bounds[bound][0])
    
    print("bounding box:", bbox)
    return bbox

def select_within_bbox(select_df, bbox):
    """
    Select features from a geopandas df that are completely within a bbox.
    The bbox coordinates are expected in the order: ['minx','miny','maxx','maxy']
    """
    selected_in_bbox_df = select_df.loc[(select_df.bounds['minx'] > bbox[0]) & 
                                   (select_df.bounds['miny'] > bbox[1]) &
                                   (select_df.bounds['maxx'] < bbox[2]) &
                                   (select_df.bounds['maxy'] < bbox[3])].copy()
    
    return selected_in_bbox_df

# calculate total area, field count, and average field size
def field_stats(input_gpd_df):
    """
    Returns a list of basic statistics (in sq km) for an input df:
    [tot_fields, tot_field_area, avg_field_area] 
    """
    tot_fields = input_gpd_df.count()[0]
    input_gpd_df['area_sqkm'] = input_gpd_df['geometry'].area/10**6
    tot_field_area = input_gpd_df['area_sqkm'].sum()
    avg_field_area = tot_field_area/tot_fields
    
    return [tot_fields, tot_field_area, avg_field_area]

def field_area(input_gpd_df):
    """
    Returns the value in sq km of the total area of the input df
    """
    # check crs
    print(input_gpd_df.crs)
    
    input_copy = input_gpd_df.copy()
    input_copy['area_sqkm'] = input_copy['geometry'].area/10**6
    tot_field_area = input_copy['area_sqkm'].sum()    
    print("area (sq km):", tot_field_area)
    
    return tot_field_area

def spatial_overlap(ref_fields, test_fields):
    """
    Compare two overlapping geodataframes.
    Returns a pandas dataframe with:
        'total area sq km', 
        'number of fields', 
        'average field size', 
        'intersect area', 
        'difference area', 
        'intersect pct', 
        'difference pct'
    """
    # get total number of fields
    ref_tot_fields = ref_fields.count()[0]
    test_tot_fields = test_fields.count()[0]
    
    # get area for each input
    ref_area_tot = field_area(ref_fields)
    test_area_tot = field_area(test_fields)
    
    # get average field area
    ref_avg_field_area = ref_area_tot/ref_tot_fields
    test_avg_field_area = test_area_tot/test_tot_fields
    
    # intersection: area contained in both geodataframes
    intersection = gpd.overlay(ref_fields, test_fields, how='intersection')
    ref_difference = gpd.overlay(ref_fields, test_fields, how='difference')
    test_difference = gpd.overlay(test_fields, ref_fields, how='difference')
    
    # calculate area for each overlap function
    intersection_area = field_area(intersection)
    ref_diff_area = field_area(ref_difference)
    test_diff_area = field_area(test_difference)
    
    # calculate percentages for each overlap function
    ref_int_pct = intersection_area/ref_area_tot
    test_int_pct = intersection_area/test_area_tot
    ref_diff_pct = ref_diff_area/ref_area_tot
    test_diff_pct = test_diff_area/test_area_tot
    
    columns = ['total area sq km', 'number of fields', 'average field size', 'intersect area', 'difference area', 'intersect pct', 'difference pct']
    ref_stats = [ref_area_tot, ref_tot_fields, ref_avg_field_area, intersection_area, ref_diff_area, ref_int_pct, ref_diff_pct]
    test_stats = [test_area_tot, test_tot_fields, test_avg_field_area, intersection_area, test_diff_area, test_int_pct, test_diff_pct]
    
    data = [ref_stats, test_stats]
    overlay_stats_df = pd.DataFrame(data, columns=columns, index=["reference","test"])
    overlay_stats_df = overlay_stats_df.transpose()

    print(overlay_stats_df)
    return overlay_stats_df

def validation_fields_comparison_func(**validation_inputs):
    ref_fp = validation_inputs['ref_fp']
    bbox_fp = validation_inputs['bbox_fp']
    test_fp = validation_inputs['test_fp']
    
    ref_df = shapefile_to_gpd_df(ref_fp)
    target_area_df = shapefile_to_gpd_df(bbox_fp)
    test_df = shapefile_to_gpd_df(test_fp)
    
    bbox_df = get_bbox(convert_crs(target_area_df))
    ref_fields_selected = select_within_bbox(convert_crs(ref_df),bbox_df)
    test_fields_selected = select_within_bbox(test_df,bbox_df)
    
    overlay_stats = spatial_overlap(ref_fields_selected, test_fields_selected)
    return overlay_stats