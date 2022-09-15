import os
import numpy as np
import torch
from torch.utils.data.dataset import IterableDataset
import rasterio
from rasterio.errors import RasterioError, RasterioIOError
from math import ceil
import pandas as pd
import random

class TrainingDataset(IterableDataset):
    """
    Dataset for training. Generates random small patches over the whole training set.
    """
    def __init__(self, 
                 dataset_csv, 
                 n_input_sources, 
                 exp_utils, 
                 data_dir,
                 control_training_set=True, 
                 n_neg_samples=None,
                 patch_size=128, 
                 verbose=False, 
                 debug=False):
        """
        Args:
            - dataset_csv (str): csv file listing the dataset inputs and target files, with one tile per row
            - n_input_sources (int): number of input sources
            - exp_utils (ExpUtils): ExpUtils object specifying methods and parameters to use for the current experiment
            - data_dir (str): directory where the data is stored
            - control_training_set (bool): whether to control the number of images without class 0 use at each epoch
            - n_neg_samples (int): number of negative samples (i.e. containing class 0 only) to use
            - patch_size (int): size of the small patches to extract from the input tiles
            - debug (bool): whether to use only a subset of the files listed in "dataset_csv", to speed up debugging
        """
        self.data_dir = data_dir
        self.n_input_sources = n_input_sources
        self.control_training_set = control_training_set
        
        self.patch_size = patch_size
        self.num_patches_per_tile = exp_utils.num_patches_per_tile
        self.exp_utils = exp_utils
        self.verbose = verbose
        self.fns = pd.read_csv(dataset_csv)
        if debug:
            self.fns = self.fns.iloc[:30]
        self._check_df_columns()
        
        self.n_fns_all = len(self.fns)

        # store filenames of positive and negative examples separately
        if control_training_set:
            self._split_fns()
            self.n_positives = len(self.fns_positives)
            self.n_negatives = len(self.fns_negatives)
            self.select_negatives(n_neg_samples)
        else:
            self.fns_positives = None
            self.fns_negatives = None
            self.n_positives = None
            self.n_negatives = None
            
    def _check_df_columns(self):
        col_names = list(self.fns)
        # target
        if 'target' not in col_names:
            raise KeyError('"target" column not found in the dataset csv file')
        # input(s)
        self._check_df_input_columns()
        # counts            
        if self.control_training_set:
            for i in range(1, self.exp_utils.n_classes):
                if 'count_{}'.format(i) not in col_names:
                    raise KeyError('Could not find count_{} column(s) in dataset csv file'.format(i))
                
    def _check_df_input_columns(self):
        col_names = list(self.fns)
        if self.n_input_sources > 1:
            self.input_col_names = ['input_{}'.format(i) for i in range(self.n_input_sources)]
            for name in self.input_col_names:
                if name not in col_names:
                    raise KeyError('Could not find {} column in dataset csv file'.format(name))
        else:
            if 'input' not in col_names:
                raise KeyError('"input" column not found in the dataset csv file')
            self.input_col_names = ['input']

    def _split_fns(self):
        """
        Creates two dataframes self.fns_positives and self.fns_negatives which store positive and negative filenames 
        separately
        """
        positive_counts = self.fns.loc[:, ['count_' + str(i) for i in range(1, self.exp_utils.n_classes)]]   
        positives_mask = positive_counts.any(axis=1).to_numpy()
        self.fns_positives = self.fns[positives_mask]
        self.fns_negatives = self.fns[~positives_mask]
        

    def select_negatives(self, n_neg_samples):
        """
        Fills self.fn with the right number of negative samples
        """
        if n_neg_samples is not None:           
            if n_neg_samples == 0: # do not use negative samples
                self.fns = self.fns_positives
            elif n_neg_samples < self.n_negatives: # pick negative samples randomly
                draw_idx = np.random.choice(self.n_negatives, size=(n_neg_samples,), replace = False)
                self.fns = pd.concat([self.fns_positives, self.fns_negatives.iloc[draw_idx]], ignore_index=True) 
                print(draw_idx)
            elif n_neg_samples >= self.n_negatives: # use all negative samples
                self.fns = pd.concat([self.fns_positives, self.fns_negatives], ignore_index=True)
            print('Using {} training samples out of {}'.format(len(self.fns), self.n_fns_all))
            
    def shuffle(self):
        self.fns = self.fns.sample(frac=1).reset_index(drop=True)

    def _get_worker_range(self, fns):
        """Get the range of tiles to be assigned to the current worker"""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        # define the range of files that will be processed by the current worker
        num_files_per_worker = ceil(len(fns) / num_workers)
        lower_idx = worker_id * num_files_per_worker
        upper_idx = min(len(fns), (worker_id+1) * num_files_per_worker)

        return lower_idx, upper_idx
    
    @staticmethod
    def seed_worker(worker_id):
        """from https://pytorch.org/docs/stable/notes/randomness.html"""
        worker_seed = torch.initial_seed() % 2**32
        # print('Worker seed {}: {}'.format(worker_id, worker_seed))
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def _stream_tile_fns(self, lower_idx, upper_idx):
        """Generator providing input and target paths tile by tile from lower_idx to upper_idx"""
        for idx in range(lower_idx, upper_idx):
            yield self.fns.iloc[idx]

    def _extract_multi_patch(self, data, scales, x, xstop, y, ystop):
        """
        Extract a patch from multisource data given the relative scales of the sources and boundary coordinates
        """
        return [self._extract_patch(data[i],scales[i], x, xstop, y, ystop) for i in range(len(data))]

    def _extract_patch(self, data, scale, x, xstop, y, ystop):
        return data[y*scale:ystop*scale, x*scale:xstop*scale]

    def _generate_patch(self, data, num_skipped_patches, coord = None):
        """
        Generates a patch from the input(s) and the targets, randomly or using top left coordinates "coord"
        Args:
            - data (list of (list of) tensors): input and target data
            - num_skipped_patches (int): current number of skipped patches (will be updated)
            - coord: top left coordinates of the patch to extract, in the coarsest modality

        Output:
            - patches (list of (list of) tensors): input and target patches
            - num_skipped_patches (int)
            - exit code (int): 0 if success, 1 if IndexError or invalid patch (due to nodata)
            - (x,y) (tuple of ints): top left coordinates of the extracted patch
        """

        input_data, target_data = data
        # find the coarsest data source
        height, width = target_data.shape[:2] 
        height = height // self.exp_utils.target_scale
        width = width // self.exp_utils.target_scale

        if coord is None: # pick the top left pixel of the patch randomly
            x = np.random.randint(0, width-self.patch_size)
            y = np.random.randint(0, height-self.patch_size)
        else: # use the provided coordinates
            x, y = coord
            
        # extract the patch
        try:
            xstop = x + self.patch_size
            ystop = y + self.patch_size
            # extract input patch
            input_patches = self._extract_multi_patch(input_data, self.exp_utils.input_scales, x, xstop, y, ystop)
            # extract target patch
            target_patch = self._extract_patch(target_data, self.exp_utils.target_scale, x, xstop, y, ystop)
        except IndexError:
            if self.verbose:
                print("Couldn't extract patch (IndexError)")
            return (None, num_skipped_patches, 1, (x, y))

        # check for no data
        skip_patch = self.exp_utils.target_nodata_check(target_patch) or self._inputs_nodata_check(input_patches) 
        if skip_patch: # the current patch is invalid
            num_skipped_patches += 1
            return (None, num_skipped_patches, 1, (x, y))

        # preprocessing (needs to be done after checking nodata)
        input_patches = self._preprocess_inputs(input_patches)
        target_patch = self.exp_utils.preprocess_training_target(target_patch)
        patches = [input_patches, target_patch]

        return (patches, num_skipped_patches, 0, (x, y))
    
    def _preprocess_inputs(self, patches):
        return self.exp_utils.preprocess_inputs(patches)
    
    def _inputs_nodata_check(self,patches):
        return self.exp_utils.inputs_nodata_check(*patches) 

    def _read_tile(self, df_row):
        try: # open files
            img_fp = [rasterio.open(os.path.join(self.data_dir, fn), "r") for fn in list(df_row[self.input_col_names])]
            target_fp = rasterio.open(os.path.join(self.data_dir, df_row['target']), "r")
        except (RasterioIOError, rasterio.errors.CRSError) as e:
            print('WARNING: {}'.format(e))
            return None

        # read data for each input source and for the targets
        try:
            img_data = [None] * self.n_input_sources
            for i, fp in enumerate(img_fp):  
                img_data[i] = np.moveaxis(fp.read(), (1, 2, 0), (0, 1, 2))
            target_data = target_fp.read(1)
        except RasterioError as e:
            print("WARNING: Error reading file, skipping to the next file")
            return None

        # close file pointers and return data
        for fp in img_fp:
            fp.close()
        target_fp.close()

        return img_data, target_data

    def _get_patches_from_tile(self, fns): 
        """Generator returning patches from one tile"""
        num_skipped_patches = 0
        # read data
        data = self._read_tile(fns)
        if data is None:
            return #skip tile if couldn't read it

        # yield patches one by one
        for _ in range(self.num_patches_per_tile):
            data_patch, num_skipped_patches, code, _ = self._generate_patch(data, num_skipped_patches, None)
            if code == 1: #IndexError or invalid patch
                continue #continue to next patch
            yield data_patch

        if num_skipped_patches>0 and self.verbose:
            print("We skipped %d patches on %s" % (num_skipped_patches, fns[0]))

    def _stream_patches(self):
        """Generator returning patches from the samples the worker calling this function is assigned to"""
        lower_idx, upper_idx = self._get_worker_range(self.fns)
        for fns in self._stream_tile_fns(lower_idx, upper_idx): #iterate over tiles assigned to the worker
            yield from self._get_patches_from_tile(fns) #generator 

    def __iter__(self):
        if self.verbose:
            print("Creating a new {} iterator").format(self.__class__.__name__)
        return iter(self._stream_patches())
    
class MultiTargetTrainingDataset(TrainingDataset):
    """
    Dataset for training. Generates random small patches over the whole training set, with input data, target data as
    well as intermediate target(s).
    """
    def __init__(self, 
                 dataset_csv, 
                 n_input_sources, 
                 n_interm_target_sources, 
                 exp_utils, 
                 data_dir,
                 control_training_set=True, 
                 patch_size=128, 
                 n_neg_samples=None, 
                 verbose=False, 
                 debug=False):
        """
        Args:
            - n_interm_target_sources: number of intermediate targets to use (i.e. number of intermediate concept 
                regression tasks)
        """
        
        if n_interm_target_sources < 1:
            raise ValueError(
                'interm_target_sources was set as {}, but should be greater than 0.'.format(n_interm_target_sources))
        self.n_interm_target_sources = n_interm_target_sources
        
        super().__init__(dataset_csv, n_input_sources, exp_utils, data_dir, control_training_set=control_training_set, 
                         n_neg_samples=n_neg_samples, patch_size=patch_size, verbose=verbose, debug=debug)
        
        
    def _check_df_columns(self):
        super()._check_df_columns()
        self._check_df_interm_target_columns()
                
    def _check_df_interm_target_columns(self):
        col_names = list(self.fns)
        self.interm_target_col_names = ['interm_target_{}'.format(i) for i in range(self.n_interm_target_sources)]
        for name in self.interm_target_col_names:
            if name not in col_names:
                raise KeyError('Could not find {} column in dataset csv file'.format(name))

    def _read_tile(self, df_row):
        img_data, target_data = super()._read_tile(df_row)
        
        try: # open files
            interm_target_fp = [rasterio.open(os.path.join(self.data_dir, fn), "r") for fn in list(df_row[self.interm_target_col_names])]
        except (RasterioIOError, rasterio.errors.CRSError) as e:
            print('WARNING: {}'.format(e))
            return None

        # read data
        try:
            interm_target_data = [None] * self.n_interm_target_sources
            for i, fp in enumerate(interm_target_fp):  
                interm_target_data[i] = fp.read(1)
        except RasterioError as e:
            print("WARNING: Error reading file, skipping to the next file")
            return None

        # close file pointers and return data
        for fp in interm_target_fp:
            fp.close()

        return img_data, target_data, interm_target_data
    
    def _generate_patch(self, data, num_skipped_patches, coord = None):
        """
        Generates a patch from the input(s) and the targets, randomly or using top left coordinates "coord"

        Args:
            - data (list of (list of) tensors): input and target data
            - num_skipped_patches (int): current number of skipped patches (will be updated)
            - coord: top left coordinates of the patch to extract, in the coarsest modality

        Output:
            - patches (list of (list of) tensors): input and target patches
            - num_skipped_patches (int)
            - exit code (int): 0 if success, 1 if IndexError or invalid patch (due to nodata)
            - (x,y) (tuple of ints): top left coordinates of the extracted patch
        """

        input_data, target_data, interm_target_data = data
        # find the coarsest data source
        height, width = target_data.shape[:2] 
        height = height // self.exp_utils.target_scale
        width = width // self.exp_utils.target_scale

        if coord is None: # pick the top left pixel of the patch randomly
            x = np.random.randint(0, width-self.patch_size)
            y = np.random.randint(0, height-self.patch_size)
        else: # use the provided coordinates
            x, y = coord
            
        # extract the patch
        try:
            xstop = x + self.patch_size
            ystop = y + self.patch_size
            # extract input patch
            input_patches = self._extract_multi_patch(input_data, self.exp_utils.input_scales, x, xstop, y, ystop)
            # extract intermediate target patch
            interm_target_patches = self._extract_multi_patch(interm_target_data, self.exp_utils.interm_target_scales, 
                                                              x, xstop, y, ystop)
            # extract target patch
            target_patch = self._extract_patch(target_data, self.exp_utils.target_scale, x, xstop, y, ystop)
        except IndexError:
            if self.verbose:
                print("Couldn't extract patch (IndexError)")
            return (None, num_skipped_patches, 1, (x, y))

        # check for no data
        skip_patch = self.exp_utils.target_nodata_check(target_patch) \
                    or self._inputs_nodata_check(input_patches) \
                    or self.exp_utils.interm_targets_nodata_check(interm_target_patches)    
        if skip_patch: # the current patch is invalid
            num_skipped_patches += 1
            return (None, num_skipped_patches, 1, (x, y))

        # preprocessing (needs to be done after checking nodata)
        input_patches = self._preprocess_inputs(input_patches)
        target_patch = self.exp_utils.preprocess_training_target(target_patch)
        interm_target_patches = [f(t, nodata_val, i) for i, (f, t, nodata_val) in enumerate(zip(
                                                                    self.exp_utils.preprocess_training_interm_targets, 
                                                                    interm_target_patches, 
                                                                    self.exp_utils.interm_target_nodata_val))]
        patches = [input_patches, target_patch, interm_target_patches]

        return (patches, num_skipped_patches, 0, (x, y))
