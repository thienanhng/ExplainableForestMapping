""" 
Creating raster format TLM labels, from 
    - a shapefile 'shp_fn' with SwissTLM3D polygons and a field 'Class' indicating the class
    - a shapefile 'img_orig_shp_fn' of tile footprints indicating the the tiles to generate
"""

import os
from tqdm import tqdm
from osgeo import gdal
from osgeo.gdalconst import GA_ReadOnly
from tqdm import tqdm

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data')
shp_fn = os.path.join(data_dir, 
                      'SwissTLM3D_edited/swissTLM3D_TLM_BODENBEDECKUNG_Wald_offen_Wald_Gebueschwald_Gehoelzflaeche.shp')
img_out_dir = os.path.join(data_dir, 'TLMRaster/5c')
img_orig_dir = os.path.join(data_dir, 'SwissImage/2017')
img_orig_shp_fn = os.path.join(data_dir, 'SwissImage/vector/ExportFootprint_2017_alti_1500_2500.shp')

prefix = "TLM5c"
downsample = 10 # 10 cm to 1m

def get_img_param(fn):
    f = gdal.Open(fn, GA_ReadOnly)
    minx, xres, _, maxy, _, yres = f.GetGeoTransform()
    npix_x = f.RasterXSize
    npix_y = f.RasterYSize
    maxx = minx + xres * npix_x
    miny = maxy + yres * npix_y
    f = None
    return minx, miny, maxx, maxy, npix_x, npix_y

def rasterize(shp_fn, img_fn, minx, miny, maxx, maxy, npix_x, npix_y, downsample = 1):
    assert npix_x % downsample == 0
    assert npix_y % downsample == 0
    npix_x_down = npix_x//downsample
    npix_y_down = npix_y//downsample
    
    command_str = "gdal_rasterize -init 0 -a 'Class' -te {} {} {} {} -ts {} {} -of 'GTiff' -ot 'Byte' {} {}"\
                .format(minx, miny, maxx, maxy, npix_x_down, npix_y_down, 
                shp_fn, img_fn)
    os.system(command_str)

for img_orig_bn in tqdm(os.listdir(img_orig_dir)):
    if os.path.splitext(img_orig_bn)[-1] == '.tif':
        tilenum = '_'.join(img_orig_bn.split('_')[2:4])
        out_fn = os.path.join(img_out_dir,"{}_{}.tif".format(prefix, tilenum))
        minx, miny, maxx, maxy, npix_x, npix_y = get_img_param(os.path.join(img_orig_dir, img_orig_bn))
        rasterize(shp_fn, out_fn, minx, miny, maxx, maxy, npix_x, npix_y, downsample = downsample)