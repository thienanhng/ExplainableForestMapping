import os
import numpy as np
import rasterio
from skimage.morphology import remove_small_holes, binary_opening
from scipy import ndimage as nd
import cv2
from tqdm import tqdm
from rasterio.windows import Window

project_dir = os.path.dirname(os.path.dirname(__file__))

def smooth_segmentation(img, area, opening=True, opening_size=3, min_area=2):
    """Post-processing of forest segmentation results"""
    values = np.unique(img)
    if len(values) > 1:
        if opening:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(opening_size,opening_size))
        for a in range(min_area, area+1): #fill holes by increasing hole size
            values = np.roll(values, 1) #to minimize the influence of the class ordering
            for val in values:
                # detect holes
                mask = img != val
                holes = mask != remove_small_holes(mask, area_threshold=a)
                if opening and (a==min_area): 
                    # initial opening step
                    mask = img == val
                    opened = binary_opening(mask, selem=kernel)
                    holes = np.logical_or(holes, opened != mask)
                # fill the holes with nearest neighbours
                ind = nd.distance_transform_edt(holes, return_distances=False, return_indices=True)
                img = img[tuple(ind)]
    return img

########################### PARAMETERS ########################################

overwrite = True                                                # whether to overwrite existing files
area = 36                                                       # maximum area of holes to fill                                                                     
min_area = area//4                                              # minimum area of holes to fill
opening_size = 3                                                # size of the opening kernel
max_margin = area                                               # size of the margin to use around tiles
exp_name = 'sb_hierarchical_MSElog1em1_MSE_doubling_negatives'  # name of the experiment that generated the tiles to process
fn_vrt = os.path.join(project_dir, 
                      'output/{}/inference/epoch_19/test_with_context/mosaic_test_with_context.vrt'.format(exp_name))

########################### PROCESSING ########################################

# open the set of tiles to process as a virtual mosaic
with rasterio.open(fn_vrt, 'r') as f_vrt:
    vrt_h = f_vrt.height
    vrt_w = f_vrt.width
    # iterate over the tiles to process
    for fn in tqdm(f_vrt.files[1:]):
        im_out_fn = fn.replace('predictions_', 'predictions_processed_a{}_k{}_'.format(area, opening_size))
        if overwrite or not os.path.exists(im_out_fn):
            with rasterio.open(fn, 'r') as f_tile:
                left, top = f_tile.bounds.left, f_tile.bounds.top
                h, w, = f_tile.height, f_tile.width
                profile = f_tile.profile
            # read the tile with a margin around its boundaries
            i_min, j_min = f_vrt.index(left, top)
            top_margin = min(max(0, i_min-max_margin), max_margin)
            left_margin = min(max(0, j_min-max_margin), max_margin)
            bottom_margin = min(max(0, vrt_h - (i_min + h+max_margin)), max_margin)
            right_margin = min(max(0, vrt_w - (j_min + w+max_margin)), max_margin)
            h_w_margin = h + top_margin + bottom_margin
            w_w_margin = w + left_margin + right_margin
            win = Window(   j_min - left_margin, 
                            i_min - top_margin, 
                            w_w_margin, 
                            h_w_margin)
            im_w_margin = f_vrt.read(1, window = win)
            # process the tile
            im_out_w_margin = smooth_segmentation(im_w_margin, 
                                                  area, 
                                                  opening=True, 
                                                  opening_size=opening_size, 
                                                  min_area=min_area)
            im_out = im_out_w_margin[top_margin:h_w_margin-bottom_margin, left_margin:w_w_margin-right_margin]
            # write the processed tile to a file
            with rasterio.open(im_out_fn, 'w', **profile) as f_out:
                f_out.write(im_out, 1)