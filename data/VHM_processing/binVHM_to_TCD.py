""" 
Computes Tree Canopy Density (TCD) rasters from a binary (i.e. thresholded) Vegetation Height Model (VHM)
"""

import os
import numpy as np
import csv
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from skimage.morphology import disk
import torch
import torch.nn.functional as f

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

################################# PARAMETERS ##################################

csv_fn = os.path.join(project_dir, "data/csv/TileKeys_val_viz_subset.csv") # tiles to process

### output directory and filename prefix
threshold_height = 1
data_dir = os.path.join(os.path.dirname(os.path.dirname(project_dir)), 'Data')
img_out_dir = os.path.join(data_dir, "TCD_{}m".format(threshold_height))
prefix_out = "TCD_{}m".format(threshold_height)

### input raster mosaic
mosaic_fn = os.path.join(
    data_dir, 
    "VHM_NFI_bin_{}m/mosaic_VHM_NFI_bin_{}m.vrt".format(threshold_height, threshold_height))


radius = 28 # radius of the averaging kernel
nodata = -1 # nodata value for pixels for which when value are missing to compute the average

################################# PROCESSING ##################################


kernel = torch.from_numpy(disk(radius)).float().unsqueeze(0).unsqueeze(0)
s = torch.sum(kernel)

with open(csv_fn, 'r') as f_csv:
    reader = csv.reader(f_csv)
    tilenum_list = list(reader)

with rasterio.open(mosaic_fn, 'r') as f_mosaic:
    # iterate over tiles
    for fn in tqdm(f_mosaic.files):
        if os.path.splitext(fn)[-1] == '.tif':
            # read the tile with margins
            with rasterio.open(fn, 'r') as f_bin:
                bb = f_bin.bounds
                profile = f_bin.profile
                nodata_mosaic = profile['nodata']
            
            i_min, j_min = f_mosaic.index(bb.left, bb.top)
            top_margin = min(max(0, i_min-radius), radius)
            left_margin = min(max(0, j_min-radius), radius)
            bottom_margin = min(max(0, f_mosaic.height - (i_min + profile['height']+radius)), radius)
            right_margin = min(max(0, f_mosaic.width - (j_min + profile['width']+radius)), radius)
            im_bin = f_mosaic.read(1, window = Window(  j_min-left_margin, 
                                                        i_min-top_margin, 
                                                        profile['width']+left_margin+right_margin, 
                                                        profile['height']+top_margin+bottom_margin))
            # check that the margins are not in nodata zones of the mosaic (zones in-between tiles)
            height, width = im_bin.shape[0], im_bin.shape[1]
            if top_margin > 0:
                nodata_rows = np.all(im_bin[:top_margin, left_margin:width-right_margin] == nodata_mosaic, axis = 1)
                if np.any(nodata_rows):
                    n_nodata_rows = np.sum(nodata_rows)
                    assert np.all(im_bin[:n_nodata_rows, left_margin:width-right_margin] == nodata_mosaic)
                    im_bin = im_bin[n_nodata_rows:, :]
                    top_margin -= n_nodata_rows
            
            if bottom_margin > 0:
                nodata_rows = np.all(im_bin[-bottom_margin:, left_margin:width-right_margin] == nodata_mosaic, axis = 1)
                if np.any(nodata_rows):
                    n_nodata_rows = np.sum(nodata_rows)
                    assert np.all(im_bin[-n_nodata_rows:, left_margin:width-right_margin] == nodata_mosaic)
                    im_bin = im_bin[:-n_nodata_rows, :]
                    bottom_margin -= n_nodata_rows
                    
            if left_margin > 0:
                nodatacols = np.all(im_bin[top_margin:height-bottom_margin, :left_margin] == nodata_mosaic, axis = 0)
                if np.any(nodatacols):
                    n_nodatacols = np.sum(nodatacols)
                    assert np.all(im_bin[top_margin:height-bottom_margin, :n_nodatacols] == nodata_mosaic)
                    im_bin = im_bin[:, n_nodatacols:]
                    left_margin -= n_nodatacols
                    
            if right_margin > 0:
                nodata_cols = np.all(im_bin[top_margin:height-bottom_margin, -right_margin:] == nodata_mosaic, axis = 0)
                if np.any(nodata_cols):
                    n_nodata_cols = np.sum(nodata_cols)
                    assert np.all(im_bin[top_margin:height-bottom_margin, -n_nodata_cols:] == nodata_mosaic)
                    im_bin = im_bin[:, :-n_nodata_cols]
                    right_margin -= n_nodata_cols
                    
                
            height, width = im_bin.shape[0], im_bin.shape[1]
            assert height >= profile['height'] and height <= profile['height'] + 2*radius
            assert width >= profile['width'] and width <= profile['width'] + 2*radius
            
            im_bin = torch.from_numpy(im_bin).float()
            im_bin[im_bin == float(nodata_mosaic)] = float('nan')
            
            # apply the spatial average
            im_out = f.conv2d(im_bin.unsqueeze(0).unsqueeze(0), kernel)/s*100
            im_out = im_out.squeeze(0).squeeze(0)
            # pad with nodata where the margin was too small
            im_out = torch.nn.functional.pad(
                                    im_out, 
                                    (radius-left_margin, radius-right_margin, radius-top_margin, radius-bottom_margin), 
                                    mode = 'constant', 
                                    value = nodata)
            # replace all values polluted by nans by nodata value
            im_out[torch.isnan(im_out)] = nodata
            # write the process tile
            fn_out = fn.replace('VHM_NFI_bin', 'TCD')
            profile['dtype'] = 'float32'
            profile['nodata'] = nodata
            with rasterio.open(fn_out, 'w', **profile) as f_out:
                f_out.write(im_out.numpy(), 1)

    
