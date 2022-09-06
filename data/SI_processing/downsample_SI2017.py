""" 
Script to downsample SwissImage imagery
"""

import os
import rasterio
from csv import reader
from tqdm import tqdm
from rasterio.enums import Resampling
from rasterio import Affine

project_dir = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(os.path.dirname(project_dir), 'Data')
orig_im_dir = os.path.join(data_dir, 'SwissImage/2017_10cm')
new_im_dir = os.path.join(data_dir, 'SwissImage/2017_25cm')
csv_fn = os.path.join(project_dir, 'data/csv/SI2017_train.csv')
resample_factor = 2.5 # 10 cm to 25 cm

with open(csv_fn, 'r') as f_csv:
    csv_reader = reader(f_csv)
    for row in tqdm(csv_reader):
        fn = row[0]
        if os.path.isfile(fn):
            if os.path.splitext(fn)[-1] == '.tif':
                fn_out = fn.replace(os.path.join(orig_im_dir, 'DOP10'), os.path.join(new_im_dir, 'DOP25'))
                if not os.path.exists(fn_out):
                    with rasterio.open(fn, 'r') as f_im:
                        output_profile = f_im.profile.copy()
                        t = f_im.profile['transform']
                        new_height, new_width = int(f_im.height // resample_factor), int(f_im.width // resample_factor)
                        img = f_im.read(out_shape = (   f_im.count, 
                                                        new_height, 
                                                        new_width),
                                                        resampling = Resampling.bilinear)
                                        
                        output_profile['height'] = new_height
                        output_profile['width'] = new_width
                        
                        output_profile['transform'] = Affine(t.a * resample_factor, t.b, t.c, t.d, t.e * resample_factor, t.f)
                        with rasterio.open(fn_out, 'w', **output_profile) as f_out:
                            f_out.write(img)

