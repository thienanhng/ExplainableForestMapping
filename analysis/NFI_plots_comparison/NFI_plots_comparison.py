"""
Sample points from a forest map to compare with NFI plots.
Samples are extracted 
    - from a raster mosaic for the model predictions
    - from a point shapefile for the TLM targets, in which the target value has already been extracted for each NFI plot 
    location (using a GIS software).
The extracted sample are saved in a .npy file and/or in a point shapefile.
"""

import sys
import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_dir)

import geopandas as gpd
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import fiona


################################# PARAMETERS ##################################

sample_from = 'raster' # 'shp'#         # whether to sample from rasters (model predictions) of a point shapefile (TLM)
save_pts = True                         # save the points in a .npy file
write_pt_shp = True                     # write the points in a copy of NFIplots_shp
source = 'bb' #'sb' #'TLM' #'sb'        # name of the field that will be added to the point shapefile
shp_dir = os.path.join(os.path.dirname(os.path.dirname(project_dir)), 'GIS', 'NFI_plots')
NFIplots_shp = os.path.join(shp_dir, 'LFI4_PLOTDATA_aoi_test_strict.shp') #shapefile containing the NFI plots
task = '' #'_ft'                        # suffix corresponding to the task to sample results from ('_ft': forest type, 
                                        # '': main task). '_ft' is used to extract the predicted forest type outside the
                                        # areas predicted as 'presence of forest'

if sample_from == 'raster': # params for raster sampling
    # virtual mosaic of the predictions to sample from
    exp_name = 'bb_hierarchical' #'sb_hierarchical_MSElog1em1_MSE_doubling_negatives' #
    vrt = os.path.join( 
            project_dir, 
            'output/{}/inference/epoch_19/test_with_context/mosaic_test_with_context_processed_a36_k3{}.vrt'.format(
                                                                                                                exp_name, 
                                                                                                                task))
    # directory where the prediction points will be saved
    dir_out_raster = os.path.join(project_dir, 
                                  'output/{}/inference/epoch_19/test_with_context'.format(exp_name)) 
else: # params for point shapefile sampling
    vals_shp = os.path.join(shp_dir, 'TLM_samples_NFIplots_test_strict.shp')
    dir_out_shp = os.path.join(project_dir, 'output/TLM_analysis') 
    

################################# SAMPLING ####################################

# build array to convert 4-classes labels/predictions to forest type
to_ft = np.full(shape=256, fill_value=-1)
to_ft[:4] = np.array([-1, 0, 1, 2]) 

if sample_from == 'raster':
    dir_out = dir_out_raster
else:
    dir_out = dir_out_shp

# create dataframe with coordinate and class of NFI plots
df = gpd.read_file(NFIplots_shp)
print(df.head())
NFI_classes = df['TLM_interp'].values
if task == '_ft':
    NFI_classes = to_ft[NFI_classes]
pred_classes = np.zeros_like(NFI_classes)
if sample_from == 'raster':
    # for each point, retrieve corresponding pixel 
    with rasterio.open(vrt, 'r') as f_vrt:
        h, w = f_vrt.height, f_vrt.width
        #profile = f_vrt.profile
        for idx in tqdm(range(df.shape[0])):
            x, y = df.iloc[idx]['geometry'].x, df.iloc[idx]['geometry'].y
            i, j = f_vrt.index(x, y)
            if i == h:
                i -= 1
            elif i > h or i < 0:
                raise IndexError('x coordinate out of file')
            if j == w:
                j -= 1
            elif j > w or j < 0:
                raise IndexError('y coordinate out of file')
            pixel = f_vrt.read(1, window=Window(j, i, 1, 1))
            pred_classes[idx] = pixel.item()
else:
    df_vals = gpd.read_file(vals_shp)
    for index, row in df.iterrows():
        # match rows by geometry
        g = row['geometry']
        pred_classes[index] = df_vals[df_vals['geometry'] == g]['class']


if save_pts:
    pts_out = np.stack((NFI_classes, pred_classes), axis=0)        
    fn_out = os.path.join(dir_out, 'NFI_comparison_pts{}.npy'.format(task))
    np.save(fn_out, pts_out)    

if write_pt_shp:
# create a point shapefile and write the samples in a new field
    new_shp_fn = os.path.join(shp_dir, '{}{}_samples_NFIplots_test_strict_with_context.shp'.format(source, task))
    with fiona.open(NFIplots_shp, 'r') as f_NFI_shp:
        meta = f_NFI_shp.meta
        meta['schema']['properties'][source] = 'int:10'
        with fiona.open(new_shp_fn, 'w', **meta) as f_new_shp:
            for idx, feature in tqdm(enumerate(f_NFI_shp)):
                sample = pred_classes[idx].item()
                feature['properties'][source] = sample
                f_new_shp.write(feature)
     