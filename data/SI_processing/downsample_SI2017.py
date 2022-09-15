""" 
Script to download + downsample SwissImage imagery
"""

import os
import rasterio
from csv import reader
from tqdm import tqdm
from rasterio.enums import Resampling
from rasterio import Affine
import argparse
import requests
import csv

def get_parser():
    parser = argparse.ArgumentParser(
        description='Downsample images from a source folder and write them in a destination folder')
    parser.add_argument('--url_csv_fn', type=str, help='Csv file where the urls of images to download are listed')
    parser.add_argument('--source_dir', type=str, help='Folder containing the original images must be written (temporarily)')
    parser.add_argument('--dest_dir', type=str, help='Folder where the downsampled images must be written')
    parser.add_argument('--delete_processed', action='store_true')
    return parser

# project_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# data_dir = os.path.join(os.path.dirname(project_dir), 'Data')
# orig_im_dir = os.path.join(data_dir, 'SwissImage/2017_10cm')
# new_im_dir = os.path.join(data_dir, 'SwissImage/2017_25cm')

def downsample_dataset(url_csv_fn, orig_im_dir, new_im_dir):
    os.makedirs(new_im_dir, exist_ok=True)
    os.makedirs(orig_im_dir, exist_ok=True)
    resample_factor = 2.5 # 10 cm to 25 cm
    with open(url_csv_fn, 'r') as f_in:
        reader = csv.reader(f_in)
        for row in tqdm(reader):
            # download image
            url = row[0]
            basename = url.rsplit('/', 1)[1]
            fn = os.path.join(orig_im_dir, basename)
            r = requests.get(url)
            open(fn, 'wb').write(r.content)
            
    #for fn in tqdm(os.listdir(orig_im_dir)):
        #if os.path.splitext(fn)[-1] == '.tif':
        
            # downsample image
            fn_out = basename.replace('swissimage-dop10_2017_', 'DOP25_LV95_')
            fn_out = fn_out.replace('_0.1_2056', '_2017_1')
            fn_out = fn_out.replace('-', '_')
            fn_out = os.path.join(new_im_dir, fn_out)
            # print('Writing downsampled image to {}'.format(fn_out))
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
            # delete original image
            os.remove(fn)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    downsample_dataset(args.url_csv_fn, args.source_dir, args.dest_dir)