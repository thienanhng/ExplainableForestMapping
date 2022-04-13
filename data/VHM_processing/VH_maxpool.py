"""
Applies a local maximum operation over the Vegetation Height Model, using a morphological dilation.
No special treatment of the tile boundaries.
"""

import rasterio
import os
import csv
from tqdm import tqdm
from skimage.morphology.grey import dilation
from skimage.morphology import disk

project_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )
kernel_size = 10
suffix = '10'    
kernel = disk(kernel_size//2)
print(kernel)

csv_fn = os.path.join(project_dir, 'data/csv/VHM_val_viz_subset.csv')
dir_out = os.path.join(os.path.dirname(os.path.dirname(project_dir)), 'Data/VHM_NFI_localmax_{}m'.format(suffix))

# get the list of files to process
with open(csv_fn, 'r') as f_csv:
    reader = csv.reader(f_csv)
    fn_list = list(reader)
    
# iterate over the list of files
for l in tqdm(fn_list):
    fn = l[0]
    # read the file
    with rasterio.open(fn, 'r') as f_in:
        profile = f_in.profile
        im = f_in.read(1)
    # process the tile and write the result
    im_out = dilation(im, kernel)
    fn_out = fn.replace('NFI', 'NFI_localmax_{}m'.format(suffix))
    with rasterio.open(fn_out, 'w', **profile) as f_out:
        f_out.write(im_out, 1)
        