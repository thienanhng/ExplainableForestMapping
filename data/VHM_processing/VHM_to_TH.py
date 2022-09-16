import os
import numpy as np
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
import torch

threshold_height = 1
prefix = "VHM_NFI"
prefix_out = "TH"
project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
data_dir = os.path.join(os.path.dirname(os.path.dirname(project_dir)), 'Data')
img_dir = os.path.join(data_dir, "VHM_NFI")
img_out_dir = os.path.join(data_dir, "TH")
mosaic_fn = os.path.join(data_dir, "VHM_NFI/mosaic.vrt")


nodata = -3.4028234663852886e+38
kernel_size = 3
margin = (kernel_size - 1) // 2
mp = torch.nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=0)

with rasterio.open(mosaic_fn, 'r') as f_mosaic:
    for fn in tqdm(f_mosaic.files):
        if os.path.splitext(fn)[-1] == '.tif':
            with rasterio.open(fn, 'r') as f_bin:
                bb = f_bin.bounds
                profile = f_bin.profile
                nodata_mosaic = profile['nodata']
            
            i_min, j_min = f_mosaic.index(bb.left, bb.top)
            top_margin = min(max(0, i_min-margin), margin)
            left_margin = min(max(0, j_min-margin), margin)
            bottom_margin = min(max(0, f_mosaic.height - (i_min + profile['height']+margin)), margin)
            right_margin = min(max(0, f_mosaic.width - (j_min + profile['width']+margin)), margin)
            im_in = f_mosaic.read(1, window = Window(  j_min-left_margin, 
                                                        i_min-top_margin, 
                                                        profile['width']+left_margin+right_margin, 
                                                        profile['height']+top_margin+bottom_margin))
            # check that the margins are not in nodata zones of the mosaic (zones in-between tiles)
            height, width = im_in.shape[0], im_in.shape[1]
            if top_margin > 0:
                nodata_rows = np.all(im_in[:top_margin, left_margin:width-right_margin] == nodata_mosaic, axis = 1)
                if np.any(nodata_rows):
                    n_nodata_rows = np.sum(nodata_rows)
                    assert np.all(im_in[:n_nodata_rows, left_margin:width-right_margin] == nodata_mosaic)
                    im_in = im_in[n_nodata_rows:, :]
                    top_margin -= n_nodata_rows
            
            if bottom_margin > 0:
                nodata_rows = np.all(im_in[-bottom_margin:, left_margin:width-right_margin] == nodata_mosaic, axis = 1)
                if np.any(nodata_rows):
                    n_nodata_rows = np.sum(nodata_rows)
                    assert np.all(im_in[-n_nodata_rows:, left_margin:width-right_margin] == nodata_mosaic)
                    im_in = im_in[:-n_nodata_rows, :]
                    bottom_margin -= n_nodata_rows
                    
            if left_margin > 0:
                nodatacols = np.all(im_in[top_margin:height-bottom_margin, :left_margin] == nodata_mosaic, axis = 0)
                if np.any(nodatacols):
                    n_nodatacols = np.sum(nodatacols)
                    assert np.all(im_in[top_margin:height-bottom_margin, :n_nodatacols] == nodata_mosaic)
                    im_in = im_in[:, n_nodatacols:]
                    left_margin -= n_nodatacols
                    
            if right_margin > 0:
                nodata_cols = np.all(im_in[top_margin:height-bottom_margin, -right_margin:] == nodata_mosaic, axis = 0)
                if np.any(nodata_cols):
                    n_nodata_cols = np.sum(nodata_cols)
                    assert np.all(im_in[top_margin:height-bottom_margin, -n_nodata_cols:] == nodata_mosaic)
                    im_in = im_in[:, :-n_nodata_cols]
                    right_margin -= n_nodata_cols
                    
            
                
            height, width = im_in.shape[0], im_in.shape[1]
            assert height >= profile['height'] and height <= profile['height'] + 2*margin
            assert width >= profile['width'] and width <= profile['width'] + 2*margin
            
            im_in = torch.from_numpy(im_in).float()
            im_in[im_in == float(nodata_mosaic)] = float('nan')
            
            # pad the images if the available margin is too small
            profile['dtype'] = 'float32'
            profile['nodata'] = nodata
            im_out = mp(im_in.unsqueeze(0).unsqueeze(0))
            im_out = im_out.squeeze(0).squeeze(0)
            # pad with nodata where the margin was too small
            im_out = torch.nn.functional.pad(
                        im_out, 
                        (margin-left_margin, margin-right_margin, margin-top_margin, margin-bottom_margin), 
                        mode = 'constant', 
                        value = nodata)
            # replace all values polluted by nans by nodata value
            im_out[torch.isnan(im_out)] = nodata
            fn_out = fn.replace(prefix, prefix_out)
            with rasterio.open(fn_out, 'w', **profile) as f_out:
                f_out.write(im_out.numpy(), 1)

    
