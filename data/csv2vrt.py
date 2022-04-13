""" 
Script to convert a set of image files listed in a csv file into a virtual mosaic
"""

from osgeo import gdal

csv_fn = 'data/csv/ALTI_val_viz_subset.csv'
with open(csv_fn, 'r') as f:
    files_list = f.read().split('\n')

vrt_fn = csv_fn.replace('csv', 'vrt')  # path to vrt to build
gdal.BuildVRT(vrt_fn, files_list)

