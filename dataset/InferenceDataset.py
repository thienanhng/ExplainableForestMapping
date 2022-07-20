import numpy as np
import torch
import rasterio
import rasterio.merge
import rasterio.transform
import rasterio.warp
from rasterio.windows import Window
import random

from torch.utils.data.dataset import Dataset

class InferenceDataset(Dataset):

    """
    Dataset for inference. 
    - Performs inference by batches of tiles
    - Reads each tile with a margin around it to avoid border effects in the predictions (the margin are removed in the 
        predictions)
    - Supports multi-source input and target rasters with different depths and resolutions (pixel size for every source 
        should be a multiple of the smallest pixel size)
    - Support several targets (main target + intermediate targets)
    """

    def __init__(self, input_vrt_fns, template_fn_df, exp_utils, tile_margin=64, batch = False, target_vrt_fn = None, 
                 interm_target_vrt_fn = None, input_nodata_val = None, target_nodata_val = None, interm_target_nodata_val = None):
        """
        Args:
            - input_vrt_fns (list of str): filenames of virtual mosaics corresponding to each input source
            - template_fn_df (Pandas dataframe): template filenames for each tile (used to save output files)
            - exp_utils (ExpUtils): ExpUtils object specifying methods and parameters to use for the current experiment
            - tile_margin (int): margin to read in neighbouring tiles
            - batch (bool): whether to use batches of several tiles
            - target_vrt_fn (str): filename of the virtual mosaic containing the segmentation target
            - interm_target_vrt_fn (list of str): filenames of virtual mosaics corresponding to each intermediate concept
                target source
            - input_nodata_val (list of int/float): nodata value for each of the input sources
            - target_nodata_val (int/float): nodata value for the target
            - interm_target_nodata_val (lsit of int/float): nodata value for each of the intermediate concept target 
                source
            
        """
        
        if isinstance(target_vrt_fn, list) or isinstance(target_vrt_fn, np.ndarray):
            target_vrt_fn = target_vrt_fn[0]
        
        # set parameters
        self.n_inputs = len(input_vrt_fns)
        self.input_vrt = [rasterio.open(fn, 'r') for fn in input_vrt_fns]
        self.input_fns = [vrt.files[1:] for vrt in self.input_vrt] # the first filename is the vrt itself
        self.n_tiles = len(self.input_fns[0])
        self.template_fn_df = template_fn_df
        self.batch = batch
        if target_vrt_fn is None:
            self.sample_target = False
            self.target_vrt = None
            self.target_fns = None
        else:
            self.sample_target = True
            self.target_vrt = rasterio.open(target_vrt_fn)
            self.target_fns = self.target_vrt.files[1:]
        if interm_target_vrt_fn is None:
            self.sample_interm_targets = False
            self.interm_target_vrt = None
            self.interm_target_fns = None
        else:
            self.sample_interm_targets = True
            self.n_interm_targets = len(interm_target_vrt_fn)
            self.interm_target_vrt = [rasterio.open(fn) for fn in interm_target_vrt_fn]
            self.interm_target_fns = [vrt.files[1:] for vrt in self.interm_target_vrt]

        self.exp_utils = exp_utils
        self.input_scales = exp_utils.input_scales
        self.target_scale = exp_utils.target_scale
        if self.sample_interm_targets:
            self.interm_target_scales = exp_utils.interm_target_scales
        self.tile_margin = tile_margin
        
        self.input_nodata_val = input_nodata_val
        self.target_nodata_val = target_nodata_val
        self.interm_target_nodata_val = interm_target_nodata_val
     
    @staticmethod   
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _get_input_nodata_mask(self, data, height, width, margins): 
        """
        Create nodata mask. A nodata pixel in the mask corresponds to an overlapping nodata pixel in any of the inputs.
        """
        # get the dimensions of the output
        top_margin, left_margin, bottom_margin, right_margin = [int(m * self.target_scale) for m in margins]
        output_height = height * self.target_scale - top_margin - bottom_margin
        output_width = width * self.target_scale - left_margin - right_margin
        check = np.full((output_height, output_width), False) 
        
        # check each input
        for i, image_data in enumerate(data):
            
            op1, _ = self.exp_utils.input_nodata_check_operator[i] 
            if self.input_nodata_val[i] is not None:
                check_orig = op1(image_data[top_margin:image_data.shape[0]-bottom_margin, 
                                left_margin:image_data.shape[1]-right_margin] == self.input_nodata_val[i], axis = -1)
                # downscale and combine with current mask
                s = self.input_scales[i] // self.target_scale
                if s == 0:
                    raise RuntimeError('At least one of the inputs is coarser that the target, this is curently not '
                                        'supported.')
                for j in range(s):
                    check = np.logical_or(check, check_orig[j::s, j::s][:output_height, :output_width])

        return check

    def _read_tile(self, vrt, fn, max_margin = None, squeeze = True, keep_margins=False):
        """ - max_margin: maximum size of the margins to read in neighbouring tiles
            - keep_margins: keep the margins even in they contain only nodata values
        """
        with rasterio.open(fn, 'r') as f_tile:
            left, top = f_tile.bounds.left, f_tile.bounds.top
            h, w, = f_tile.height, f_tile.width
        i_min, j_min = vrt.index(left, top)
        top_max_margin, left_max_margin, bottom_max_margin, right_max_margin = max_margin
        if max_margin is not None:
            # compute available margins around the tile
            top_margin = int(min(max(0, i_min-top_max_margin), top_max_margin))
            left_margin = int(min(max(0, j_min-left_max_margin), left_max_margin))
            bottom_margin = int(min(max(0, vrt.height - (i_min + h + bottom_max_margin)), bottom_max_margin))
            right_margin = int(min(max(0, vrt.width - (j_min + w + right_max_margin)), right_max_margin))
        else:
            top_margin, left_margin, bottom_margin, right_margin = 0, 0, 0, 0
        # read the tile + margins
        win = Window(   j_min - left_margin, 
                        i_min - top_margin, 
                        w + left_margin + right_margin, 
                        h + top_margin + bottom_margin)

        data = vrt.read(window = win)
        
        # remove margins if they only contain nodata
        if not keep_margins:
            if left_margin > 0:
                if np.all(data[:, :, :left_margin] == vrt.nodata):
                    data = data[:, :, left_margin:]
                    left_margin = 0
            if top_margin > 0:
                if np.all(data[:, :top_margin, :] == vrt.nodata):
                    data = data[:, top_margin:, :]
                    top_margin = 0
            if right_margin > 0:
                if np.all(data[:, :, -right_margin:] == vrt.nodata):
                    data = data[:, :, :-right_margin]
                    right_margin = 0
            if bottom_margin > 0:
                if np.all(data[:, -bottom_margin:, :] == vrt.nodata):
                    data = data[:, :-bottom_margin, :] 
                    bottom_margin = 0
                
        if self.batch:
            # pad the margins so that batch elements all have the same size
            # margins now mean "margins with valid data"
            left_pad = int(left_max_margin - left_margin)               
            top_pad = int(top_max_margin - top_margin)
            right_pad = int(right_max_margin - right_margin)
            bottom_pad = int(bottom_max_margin - bottom_margin)
            data = np.pad(data, 
                          ((0, 0), (top_pad, bottom_pad), (left_pad, right_pad)), 
                          mode='constant', 
                          constant_values=0)
                
        if data.shape[0] == 1 and squeeze:
            data = data.squeeze(0)
        else:
            data = np.moveaxis(data, (1, 2, 0), (0, 1, 2))
        return data, (h + top_margin + bottom_margin, w + left_margin + right_margin), \
                (top_margin, left_margin, bottom_margin, right_margin)

    def __getitem__(self, idx):

        #### read tiles
        tile_margins = None
        
        image_data = [None] * self.n_inputs
        for i in range(self.n_inputs):
            s = self.input_scales[i]
            if tile_margins is None:
                tile_margins = [self.tile_margin] * 4
            max_margin = [self.tile_margin * s] * 4 if self.batch else [m*s for m in tile_margins]
            data, (height, width), margins = self._read_tile(self.input_vrt[i], 
                                                            self.input_fns[i][idx], 
                                                            max_margin=max_margin,
                                                            squeeze = False)
            # check the size of the image
            height_i = height // s
            width_i = width // s
            margins_i = [m/s for m in margins]
            if i == 0:
                tile_height, tile_width = height_i, width_i
                tile_margins = margins_i
            else:
                if height_i != tile_height or width_i != tile_width or (margins_i != tile_margins and not self.batch):
                    raise RuntimeError('The dimensions of the input sources do not match: '
                                        '(height={}, width={}, margins={}) for the first source '
                                        'v.s (height={}, width={}, margins={}) for the {}th source'
                                        .format(tile_height, tile_width, ','.join([str(t) for t in tile_margins]),
                                                height_i, width_i, ','.join([str(t) for t in margins_i]), i+1))
            image_data[i] = data
        
            
        if self.sample_target:
            s = self.target_scale
            # use the same margins as in the input(s)
            max_margin = [m*s for m in tile_margins] 
            target_data, (target_height, target_width), margins = self._read_tile(self.target_vrt, 
                                                self.target_fns[idx], 
                                                max_margin=max_margin, keep_margins=True)
            height_i = target_height // s
            width_i = target_width // s
            margins_i = [m/s for m in margins]
            if height_i != tile_height or width_i != tile_width or (margins_i != tile_margins and not self.batch):
                    raise RuntimeError('The dimensions of the inputs and targets do not match: '
                                        '(height={}, width={}, margins={}) for the inputs '
                                        'v.s (height={}, width={}, margins={}) for the target'
                                        .format(tile_height, tile_width, ','.join([str(t) for t in tile_margins]),
                                                height_i, width_i, ','.join([str(t) for t in margins_i])))
        else:
            target_data = None
        
        if self.sample_interm_targets:
            interm_target_data = [None] * self.n_interm_targets
            for i in range(self.n_interm_targets):
                s = self.interm_target_scales[i]
                # use the same margins as in the input(s)
                max_margin = [m*s for m in tile_margins] 
                data, (interm_target_height, interm_target_width), margins = self._read_tile(self.interm_target_vrt[i], 
                                                                self.interm_target_fns[i][idx], 
                                                                max_margin=max_margin, keep_margins=True)
                # check the size of the image
                height_i = interm_target_height // s
                width_i = interm_target_width // s
                margins_i = [m/s for m in margins]
                if height_i != tile_height or width_i != tile_width or (margins_i != tile_margins and not self.batch):
                    raise RuntimeError('The dimensions of the sources do not match: '
                                        '(height={}, width={}, margins={}) for the first input source '
                                        'v.s (height={}, width={}, margins={}) for the {}th regression target source'
                                        .format(tile_height, tile_width, ','.join([str(t) for t in tile_margins]),
                                                height_i, width_i, ','.join([str(t) for t in margins_i]), i))
                interm_target_data[i] = data
        else:
            interm_target_data = None
              
        input_nodata_mask = self._get_input_nodata_mask(image_data, tile_height, tile_width, tile_margins)
            
        # preprocess input
        inputs = [None] * self.n_inputs
        for i in range(self.n_inputs):
            inputs[i] = self.exp_utils.preprocess_input(image_data[i], i) 

        # preprocess target
        if self.sample_target:
            # target data to compute loss
            targets = self.exp_utils.preprocess_training_target(target_data) 
            # target data for inference metrics
            s = self.target_scale
            if self.batch:
                top_margin, left_margin, bottom_margin, right_margin = [int(self.tile_margin * s)] * 4
            else:
                top_margin, left_margin, bottom_margin, right_margin = [int(m * s) for m in tile_margins]
            target_tile = self.exp_utils.preprocess_inference_target(target_data[top_margin:target_data.shape[-2]-bottom_margin, 
                                                                                left_margin:target_data.shape[-1]-right_margin])
                                                                                                                     
        else:
            targets = torch.tensor([]) # empty tensor instead of None to avoid error with collate_fn
            target_tile = torch.tensor([])
            
        # preprocess intermediate targets
        if self.sample_interm_targets:
            interm_targets = [None] * self.n_interm_targets
            interm_target_tiles = [None] * self.n_interm_targets
            for i in range(self.n_interm_targets):
                # intermediate targets to compute loss
                interm_targets[i] = self.exp_utils.preprocess_training_interm_targets[i](
                                                                                        interm_target_data[i], 
                                                                                        self.interm_target_nodata_val[i], 
                                                                                        i) 
                # intermediate targets for inference metrics
                s = self.interm_target_scales[i]
                if self.batch:
                    top_margin, left_margin, bottom_margin, right_margin = [int(self.tile_margin * s)] * 4
                else:
                    top_margin, left_margin, bottom_margin, right_margin = [int(m * s) for m in tile_margins]
                interm_target_tiles[i] = self.exp_utils.preprocess_inference_interm_targets[i](
                                                        interm_target_data[i][
                                                                top_margin:interm_target_data[i].shape[-2]-bottom_margin, 
                                                                left_margin:interm_target_data[i].shape[-1]-right_margin], 
                                                        i)
            
        else:
            interm_targets = [] # empty list instead of None to avoid error with collate_fn
            interm_target_tiles = []

        tile_margins = torch.tensor([int(m) for m in tile_margins])
        return (inputs, targets, interm_targets), \
                (target_tile, interm_target_tiles), \
                tile_margins, input_nodata_mask, self.template_fn_df.iloc[idx]
          

    def __len__(self):
        return self.n_tiles
    
    def __del__(self):
        for vrt in self.input_vrt:
            vrt.close()
        if self.target_vrt is not None:
            self.target_vrt.close()
        if self.interm_target_vrt is not None:
            for vrt in self.interm_target_vrt:
                vrt.close()

    





    

    

