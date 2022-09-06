import os
import rasterio
import shutil
import random
from osgeo import gdal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .eval_utils import cm2rates, rates2metrics, my_confusion_matrix, get_regr_error_map, get_mae, get_seg_error_map
from dataset import InferenceDataset
from utils.ExpUtils import I_NODATA_VAL, F_NODATA_VAL
from .write_utils import Writer
from tqdm import tqdm

class Inference():
    """
    Class to perform inference and evaluate predictions on a set of samples. If used for validation during training, 
    the class must be instantiated once before training and Inference.infer() can be called at each epoch.
    Virtual mosaics with all the tiles for each source are created so that the Dataset can sample patches that overlap 
    several neighboring tiles. If they exist, the nodata values of the inputs rasters are used to fill the gaps, 
    otherwise a new nodata value is introduced depending on the raster data type. When calling infer(), the criteria 
    ignore_index attributes are modified accordingly.
    """
    def __init__(self, model, file_list, exp_utils, padding=64, tile_margin=64, output_dir=None, evaluate=True, 
                save_hard=True, save_soft=True, save_error_map=False, save_corr=False, save_interm=False,
                batch_size=1, num_workers=0, device=0, undersample=1, decision='f', random_seed=None,
                weight_bin_loss=False):

        """
        Args:
            - model (nn.Module): model to perform inference with
            - file_list (str): csv file containing the files to perform inference on (1 sample per row)
            - exp_utils (ExpUtils): object containing information of the experiment/dataset
            - padding (int): margin to remove around predictions
            - tile_margin (int): margin to read around each tile to provide additional context (should be equal or 
                smaller than padding)
            - output_dir (str): directory where to write output files
            - evaluate (bool): whether to evaluate the predictions
            - save_hard (bool): whether to write hard predictions into image files
            - save_soft (bool): whether to write soft predictions into image files
            - save_error_map (bool): whether to write an error map between hard predictions and targets into image files
            - save_corr (bool): whether to write correction activations (semantic bottleneck model) and a change map 
                (before vs. after correction) into image files
            - save_interm (bool): whether to write intermediate predictions (semantic bottleneck model) into image files
            - batch_size (int): batch size
            - num_workers (int): number of workers to use for the DataLoader. Recommended value is 0 because the tiles
                are processed one by one
            - device (torch.device): device to use to perform inference 
            - undersample (int): undersampling factor to reduction the size of the dataset. Example: if undersample = 100, 
                1/100th of the dataset's samples will be randomly picked to perform inference on.
            - random_seed (int): random seed for Pytorch
            - weight_bin_loss (bool): weight the binary forest presence/absence loss with weights from the class 
            of the full set of classes. 
        """

        self.evaluate = evaluate  
        self.save_hard = save_hard
        self.save_soft = save_soft
        self.save_corr = save_corr
        self.save_interm = save_interm
        self.save_error_map = save_error_map
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.undersample = undersample
        self.exp_utils = exp_utils
        self.binary_map = self.exp_utils.decision_func_2 is not None # generate binary map
        self.padding = padding
        self.tile_margin = tile_margin
        self.decision = decision
        self.weight_bin_loss = weight_bin_loss
        self.input_vrt_fn = None # used to indicate that virtual raster mosaic(s) has not been created yet
        
        g = torch.Generator()
        if random_seed is not None:
            g.manual_seed(random_seed)
        self.g = g

        self.n_inputs = self.exp_utils.n_input_sources
        self.sem_bot = self.exp_utils.sem_bot
        if self.sem_bot:
            self.n_interm_targets = len(exp_utils.interm_target_sources)

        # create a temporary directory to save virtual raster mosaics
        self.tmp_dir = 'tmp'
        i = 0
        while os.path.isdir(self.tmp_dir):
            i += 1
            self.tmp_dir = 'tmp_{}'.format(i)
        os.mkdir(self.tmp_dir)           
        # create the column strings to read the dataframe
        self._get_col_names()

        file_list_ext = os.path.splitext(file_list)[-1]
        if file_list_ext == '.csv':
            df = pd.read_csv(file_list)
        else:
            raise ValueError('file_list should be a csv file ("*.csv")')
        if evaluate or save_error_map:
            if 'target' not in df:
                raise RuntimeError('"target" column must be provided in the file list to compute error maps and metrics')
        self.n_samples = len(df)
        self._fns_df = df # self._fns_df should not be modified

        # Define output normalization (logits -> probabilities) functions
        if self.decision == 'f':
            self.seg_normalization = nn.Softmax(dim = 1)
        else:
            self.seg_normalization = self._normalize_hierarchical_output
        # Initialize cumulative confusion matrix
        self.cum_cms = {}
        if self.evaluate:
            self.cum_cms['seg'] = np.empty((self.exp_utils.n_classes,) * 2)
            if self.binary_map:
                self.cum_cms['seg_2'] = np.empty((self.exp_utils.n_classes_2,) * 2)
                if self.decision == 'h':
                    self.cum_cms['seg_1'] = np.empty((self.exp_utils.n_classes_1,) * 2)
            if self.sem_bot:
                self.cum_cms['seg_rule'] = np.empty((self.exp_utils.n_classes,) * 2)
                if self.binary_map:
                    self.cum_cms['seg_rule_2'] = np.empty((self.exp_utils.n_classes_2,) * 2)
                    if self.decision == 'h':
                        self.cum_cms['seg_rule_1'] = np.empty((self.exp_utils.n_classes_1,) * 2)
                for i in range(self.n_interm_targets):
                    key = 'regr_{}'.format(i)
                    self.cum_cms[key] = np.empty((len(self.exp_utils.thresholds[i]) + 1,) * 2)
                      

    def _normalize_hierarchical_output(self, output):
        output_1 = output[:, :self.exp_utils.n_classes_1]
        output_2 = output[:, self.exp_utils.n_classes_1:]
        norm_output_1 = torch.softmax(output_1, dim = 1)
        norm_output_2 = torch.sigmoid(output_2)
        return torch.cat((norm_output_1, norm_output_2), dim = 1)
        
    def _get_col_names(self):
        """Get the column names used to read the dataset dataframe"""
        if self.n_inputs < 2:
            self.input_col_names = ['input']
        else:
            self.input_col_names = ['input_' + str(i) for i in range(self.n_inputs)]
        if self.sem_bot:
            if self.evaluate or self.save_error_map:
                if self.n_interm_targets < 2:
                    self.interm_target_col_names = 'interm_target'
                else:
                    self.interm_target_col_names = ['interm_target_' + str(i) for i in range(self.n_interm_targets)]
    
    def _get_vrt_from_df(self, df):
        """Build virtual mosaic rasters from files listed in dataframe df"""
        #### inputs ###########################################################
        self.input_vrt_fns = [None]*self.n_inputs
        self.input_vrt_nodata_val = [None]*self.n_inputs
        for i, col_name in enumerate(self.input_col_names):
            fns = df[col_name]
            vrt_fn = os.path.join(self.tmp_dir, '{}.vrt'.format(col_name))
            if self.exp_utils.input_nodata_val[i] is None:
                # read the first tile just to know the data type:
                with rasterio.open(fns[0], 'r') as f_tile:
                    dtype = f_tile.profile['dtype']
                if dtype == 'uint8':
                    self.input_vrt_nodata_val[i] = I_NODATA_VAL
                elif dtype.startswith('uint'):
                    self.input_vrt_nodata_val[i] = I_NODATA_VAL
                    print('WARNING: nodata value for {} set to {}'.format(col_name, I_NODATA_VAL))
                else:
                    # the min and max float32 values are not handled by GDAL, using value -1 instead
                    self.input_vrt_nodata_val[i] = F_NODATA_VAL
                    print('WARNING: nodata value for {} set to {}'.format(col_name, F_NODATA_VAL)) 
            else:
                self.input_vrt_nodata_val[i] = self.exp_utils.input_nodata_val[i]
            gdal.BuildVRT(  vrt_fn, 
                            list(fns),
                            VRTNodata=self.input_vrt_nodata_val[i],
                            options = ['overwrite'])   
            self.input_vrt_fns[i] = vrt_fn
        
        self.target_vrt_fn = None
        self.target_vrt_nodata_val = None
        self.interm_target_vrt_fns = None
        self.interm_target_vrt_nodata_val = None
        if self.evaluate or self.save_error_map:
            #### main target ##################################################
            fns = df['target']  
            self.target_vrt_fn = os.path.join(self.tmp_dir, 'target.vrt') 
            if self.exp_utils.target_nodata_val is None:
                # read the tile just to know the data type:
                with rasterio.open(fns[0], 'r') as f_tile:
                    dtype = f_tile.profile['dtype']
                if dtype == 'uint8':
                    self.target_vrt_nodata_val = I_NODATA_VAL
                else:
                    raise ValueError('The main target should be of type uint8, found {} instead'.format(dtype))
            else:
                self.target_vrt_nodata_val = self.exp_utils.target_nodata_val
            gdal.BuildVRT(  self.target_vrt_fn, 
                            list(fns),
                            VRTNodata=self.target_vrt_nodata_val,
                            options = ['overwrite'])
            #### intermediate targets #########################################
            if self.sem_bot:
                if self.evaluate or self.save_error_map:
                    self.interm_target_vrt_fns = [None]*self.n_interm_targets
                    self.interm_target_vrt_nodata_val = [None]*self.n_interm_targets
                    for i, col_name in enumerate(self.interm_target_col_names):
                        fns = df[col_name]
                        vrt_fn = os.path.join(self.tmp_dir, '{}.vrt'.format(col_name))
                        if self.exp_utils.interm_target_nodata_val[i] is None:
                            # read the first tile just to know the data type:
                            with rasterio.open(fns[0], 'r') as f_tile:
                                dtype = f_tile.profile['dtype']
                            if dtype == 'uint8':
                                self.interm_target_vrt_nodata_val[i] = I_NODATA_VAL
                            elif dtype.startswith('uint'):
                                self.interm_target_vrt_nodata_val[i] = I_NODATA_VAL
                                print('WARNING: nodata value for {} set to {}'.format(I_NODATA_VAL))
                            else:
                                self.interm_target_vrt_nodata_val[i] = F_NODATA_VAL
                                print('WARNING: nodata value for {} set to {}'.format(col_name, F_NODATA_VAL))
                        else:
                            self.interm_target_vrt_nodata_val[i] = self.exp_utils.interm_target_nodata_val[i]
                        
                        gdal.BuildVRT(  vrt_fn, 
                                        list(fns),
                                        VRTNodata=self.interm_target_vrt_nodata_val[i],
                                        options = ['overwrite']) 
                        
                        self.interm_target_vrt_fns[i] = vrt_fn
            

    def _select_samples(self):
        """Select samples to perform inference on"""
        # use a random subset of the data
        idx = random.sample(range(self.n_samples), self.n_samples//self.undersample)
        df = self._fns_df.iloc[idx]
        return df.reset_index(drop = True)

    def _reset_cm(self):
        """Reset the confusion matrix/matrices with zeros"""
        if self.evaluate:
            for key in self.cum_cms:
                self.cum_cms[key].fill(0)
                

    def _get_decisions(self, actv, target_data, rule_actv=None, interm_actv=None, interm_target_data=None, nodata_mask=None):
        """Obtain decisions from soft outputs (argmax) and update confusion matrix/matrices"""
        # define main and binary outputs/targets and compute hard predictions
        if self.decision == 'f':
            # define the outputs 
            output = actv
            output_2 = actv
            output_rule = rule_actv
            output_rule_2 = rule_actv
            # compute hard predictions
            output_hard = self.exp_utils.decision_func(output)
            output_hard_1 = None
            if rule_actv is not None:
                output_hard_rule = self.exp_utils.rule_decision_func(output_rule)
            else:
                output_hard_rule = None
            # define the targets 
            output_hard_2 = None
            if self.evaluate or self.binary_map: 
                # get naive binary output
                output_hard_2 = self.exp_utils.decision_func_2(output_2)
                if self.evaluate:
                    target = target_data
                    target_2 = self.exp_utils.target_recombination(target_data)
                    #if self.exp_utils.decision_func_2 is not None:
                    if self.sem_bot:
                        rule_target = target_data
                if self.sem_bot:
                    output_hard_rule_2 = self.exp_utils.rule_decision_func_2(output_rule_2)
         
        else:
            # define the outputs 
            output_1 = actv[:, :-1]
            output_2 = actv[:, -1]
            if rule_actv is not None:
                output_rule_1, output_rule_2 = rule_actv[:, :-1], rule_actv[:, -1]
            # compute hard predictions
            output_hard_1 = self.exp_utils.decision_func(output_1) # ForestType
            output_hard_2 = self.exp_utils.decision_func_2(output_2) # forest presence/absence
            output_hard = (output_hard_1 + 1) * output_hard_2 # apply decision tree -> 4 classes
            if rule_actv is not None:
                output_hard_rule_1 = self.exp_utils.rule_decision_func(output_rule_1)
                output_hard_rule_2 = self.exp_utils.rule_decision_func_2(output_rule_2)
                output_hard_rule = (output_hard_rule_1 + 1) * output_hard_rule_2
            else:
                output_hard_rule = None
            # define the targets
            if self.evaluate or self.save_error_map:
                target = target_data[:, -1] # 4 classes
                target_1 = target_data[:, 0] # ForestType
                if output_hard_2 is not None:
                    target_2 = target_data[:, 1] # forest presence/absence
                if output_hard_rule is not None:
                    rule_target = target_data[:, -1] # 4 classes
        if self.sem_bot:
            output_hard_regr = [None] * len(interm_actv)
            for i, t in enumerate(self.exp_utils.unprocessed_thresholds):
                rep_thresh = np.tile(t[:, np.newaxis, np.newaxis, np.newaxis], (1, *interm_actv[i].shape))
                output_hard_regr[i] = np.sum(interm_actv[i] > rep_thresh, axis = 0) 
        else:
            output_hard_regr = None
                    
        # apply nodata value to invalid pixels (must be done before computing the confusion matrices)
        if nodata_mask is not None:
            output_hard[nodata_mask] = self.exp_utils.i_out_nodata_val
            if output_hard_1 is not None:
                output_hard_1[nodata_mask] = self.exp_utils.i_out_nodata_val
            if output_hard_2 is not None:
                output_hard_2[nodata_mask] = self.exp_utils.i_out_nodata_val   
            if self.sem_bot:
                for i in range(len(output_hard_regr)):
                    output_hard_regr[i][nodata_mask] =  self.exp_utils.i_out_nodata_val 
            if rule_actv is not None:
                output_hard_rule[nodata_mask] = self.exp_utils.i_out_nodata_val
                output_hard_rule_1[nodata_mask] = self.exp_utils.i_out_nodata_val
                output_hard_rule_2[nodata_mask] = self.exp_utils.i_out_nodata_val       
                
        ########## update confusion matrices #########
        # main task
        if self.evaluate:
            self.cum_cms['seg']+= my_confusion_matrix(target, 
                                                     output_hard,
                                                     self.exp_utils.n_classes)
            # other tasks / output
            if self.binary_map:
                self.cum_cms['seg_2'] += my_confusion_matrix(
                                            target_2, 
                                            output_hard_2, self.exp_utils.n_classes_2)
                if self.decision == 'h':
                    if self.evaluate:
                        self.cum_cms['seg_1'] += my_confusion_matrix(
                                                target_1, 
                                                output_hard_1, self.exp_utils.n_classes_1)

            if self.sem_bot:
                if output_hard_rule is not None:
                    self.cum_cms['seg_rule'] += my_confusion_matrix(rule_target, 
                                                                   output_hard_rule, 
                                                                   self.exp_utils.n_classes)
                    if self.binary_map:
                        # if self.decision == 'f'
                        self.cum_cms['seg_rule_2'] += my_confusion_matrix(
                                                    target_2, 
                                                    output_hard_rule_2, self.exp_utils.n_classes_2)
                        if self.decision == 'h':
                            if self.evaluate:
                                self.cum_cms['seg_rule_1'] += my_confusion_matrix(
                                                        target_1,  
                                                        output_hard_rule_1, self.exp_utils.n_classes_1)
                if self.decision == 'h':
                    if self.evaluate:
                        self.cum_cms['seg_1'] += my_confusion_matrix(
                                                target_1, 
                                                output_hard_1, self.exp_utils.n_classes_1)
                for i in range(self.n_interm_targets):
                    if output_hard_regr[i] is not None:
                        mask = np.ravel(interm_target_data[i]) != self.exp_utils.interm_target_nodata_val[i]
                        rep_thresh = np.tile(self.exp_utils.unprocessed_thresholds[i][:, np.newaxis], 
                                            (1, np.sum(mask))) # this is computed twice
                        # transform continuous target into categories
                        target_cat = np.sum(np.ravel(interm_target_data[i])[mask] >= rep_thresh, axis = 0)
                        self.cum_cms['regr_{}'.format(i)] += my_confusion_matrix(
                                                                            target_cat, 
                                                                            np.ravel(output_hard_regr[i])[mask], 
                                                                            len(self.exp_utils.thresholds[i]) + 1)

        return output_hard, output_hard_2, output_hard_1, output_hard_rule, output_hard_regr

    def _compute_metrics(self):
        """Compute classification metrics from confusion matrices"""
        reports = {}
        for key in self.cum_cms:
            reports[key] = rates2metrics(cm2rates(self.cum_cms[key]), self.exp_utils.class_names[key])
        return reports

    def _infer_sample(self, batch_data, batch_margins, 
                      seg_criterion = None, seg_criterion_2 = None,
                      regr_criteria = None, correction_penalizer = None):
        """Performs inference on one batch."""

        # forward pass
        inputs, targets, interm_targets = batch_data 
        input_data = [data.contiguous().to(self.device) for data in inputs] 
        if targets.nelement() > 0: #targets is not None:
            target_data = targets.to(self.device) 
        if len(interm_targets) > 0: #interm_targets is not None:
            interm_target_data = [data.contiguous().to(self.device) for data in interm_targets] 
        with torch.no_grad():
            # forward pass
            if self.sem_bot:
                t_main_actv, t_rule_categories, t_corr_actv, t_interm_actv = self.model(*input_data)
            else:
                t_main_actv = self.model(*input_data)
        
        # compute losses
        seg_loss, valid_px = None, None
        seg_bin_loss, valid_bin_px = None, None
        if self.sem_bot:                                                                    
            regr_loss, regr_weights = None, None
            corr_loss, valid_corr_px = None, None        
        if self.evaluate:
            if seg_criterion is not None:
                if seg_criterion_2 is not None:
                    seg_actv, bin_seg_actv = t_main_actv[:, :-1], t_main_actv[:, -1]
                    seg_target, bin_seg_target = target_data[:, 0], target_data[:, 1].float() # BCE loss needs float
                    # compute validation loss for binary subtask (last two channels)
                    bin_seg_mask = bin_seg_target != self.target_vrt_nodata_val # custom ignore_index
                    if self.weight_bin_loss:
                        seg_bin_loss = seg_criterion_2(
                            bin_seg_actv[bin_seg_mask], 
                            bin_seg_target[bin_seg_mask], 
                            torch.clamp((seg_target[bin_seg_mask]) + 1 * bin_seg_target[bin_seg_mask], min=None, max=255)).item()
                    else:
                        seg_bin_loss = seg_criterion_2(bin_seg_actv[bin_seg_mask], bin_seg_target[bin_seg_mask]).item()
                    valid_bin_px = torch.sum(bin_seg_mask).item()
                else:
                    seg_actv = t_main_actv
                    seg_target = target_data
                # main loss
                seg_mask = seg_target != seg_criterion.ignore_index
                seg_loss = seg_criterion(seg_actv, seg_target).item()
                valid_px = torch.sum(seg_mask).item()
            if self.sem_bot:
                if regr_criteria is not None:
                    regr_loss = [0] * len(regr_criteria)
                    regr_weights = [0] * len(regr_criteria)
                    for i in range(len(regr_criteria)):
                        rl, rw = regr_criteria[i](t_interm_actv[:,i], 
                                                    interm_target_data[i])
                        regr_loss[i], regr_weights[i] = rl.item(), rw.item()
                if correction_penalizer is not None:
                    # biased estimation because of the margins
                    corr_loss = correction_penalizer(t_corr_actv).item()
                    valid_corr_px = t_corr_actv.shape[0]
        
        # move predictions to cpu
        main_pred = self.seg_normalization(t_main_actv).cpu()
        if self.sem_bot:
            rule_pred = self.exp_utils.prob_encoding[t_rule_categories.cpu()].movedim((0, 3, 1, 2), (0, 1, 2, 3))
            corr_actv = t_corr_actv.cpu()
            interm_pred = t_interm_actv.cpu()
                
        s = self.exp_utils.target_scale
        padding = self.padding*s         
            
        height, width = main_pred.shape[-2:]
        if self.batch_size == 1:
            # remove margins
            top_margin, left_margin, bottom_margin, right_margin = batch_margins[0]
            main_pred = main_pred[:, :, top_margin:height-bottom_margin, left_margin:width-right_margin]
            if self.sem_bot:
                corr_actv = corr_actv[:, :, top_margin:height-bottom_margin, left_margin:width-right_margin]
                rule_pred = rule_pred[:, :, top_margin:height-bottom_margin, left_margin:width-right_margin]
                interm_pred = interm_pred[:, :, top_margin:height-bottom_margin, left_margin:width-right_margin]
        else:
            # remove fixed margins
            margin = self.tile_margin * s
            main_pred = main_pred[:, :, margin:-margin, margin:-margin] 
            if self.sem_bot:
                corr_actv = corr_actv[:, :, margin:height-margin, margin:width-margin]
                rule_pred = rule_pred[:, :, margin:height-margin, margin:width-margin]
                interm_pred = interm_pred[:, :, margin:height-margin, margin:width-margin]   
                 
        height, width = main_pred.shape[-2:]    
        nopred_mask = torch.ones(self.batch_size, height, width)
        
        top_nopred, left_nopred, bottom_nopred, right_nopred = padding - (batch_margins.T * s)
        current_batch_size = len(inputs[0])
        h_range = torch.arange(height).unsqueeze(-1).expand(height, current_batch_size)
        w_range = torch.arange(width).unsqueeze(-1).expand(width, current_batch_size)
        x_mask = torch.logical_and(h_range >= top_nopred, h_range < height - bottom_nopred).T
        y_mask = torch.logical_and(w_range >= left_nopred, w_range < width - right_nopred).T
        nopred_mask = ~torch.einsum('bi,bj->bij', (x_mask, y_mask))
        main_pred[nopred_mask.unsqueeze(1).expand(current_batch_size, main_pred.shape[1], height, width)] = 0 
        if self.sem_bot:
            corr_actv[nopred_mask.unsqueeze(1).expand(current_batch_size, corr_actv.shape[1], height, width)] = 0
            rule_pred[nopred_mask.unsqueeze(1).expand(current_batch_size, rule_pred.shape[1], height, width)] = 0
            interm_pred[nopred_mask.unsqueeze(1).expand(current_batch_size, interm_pred.shape[1], height, width)] = 0
                
        if self.sem_bot:
            return (main_pred, rule_pred, corr_actv, interm_pred), \
                    ((seg_loss, valid_px), (seg_bin_loss, valid_bin_px), \
                        (regr_loss, regr_weights), (corr_loss, valid_corr_px)), nopred_mask
        else:
            return main_pred, ((seg_loss, valid_px), (seg_bin_loss, valid_bin_px)), nopred_mask        
                

    def infer(self, seg_criterion=None, seg_criterion_2=None, regr_criteria=None, correction_penalizer=None, 
                regr_pts_per_tile = 200, detailed_regr_metrics = False):
        """
        Perform tile by tile inference on a dataset, evaluate and save outputs if needed

        Args:
            - criterion (nn.Module): criterion used for training, to be evaluated at validation as well to track 
                    overfitting
            - criterion_2 (nn.Module): criterion used for training the binary task (in case of hierarchical tasks structure)
            - regr_criteria (nn.Module): critera used for training the regression tasks
            - correction_penalizer (nn.Module): criteria used to penalize the correctiona activations during training
            - regr_pts_per_tile (int): number of random regression values to store per tile (useful for scatterplot)
            - detialed_regr_metrics (bool): whether to compute R2 and RMSE scores for the regression tasks
        """
        self.model.eval()
        
        if self.undersample > 1 or self.input_vrt_fn is None:
            # select sample to perform inference on
            df = self._select_samples()
            # create virtual mosaics (and set nodata values)
            self._get_vrt_from_df(df)
        
        #create dataset
        ds = InferenceDataset(self.input_vrt_fns, 
                              template_fn_df = df.iloc[:, 0], # use first column (first input source) as template filename
                              exp_utils=self.exp_utils, 
                              tile_margin = self.tile_margin, 
                              batch = self.batch_size > 1,
                              target_vrt_fn = self.target_vrt_fn,
                              interm_target_vrt_fn= self.interm_target_vrt_fns,
                              input_nodata_val = self.input_vrt_nodata_val,
                              target_nodata_val = self.target_vrt_nodata_val,
                              interm_target_nodata_val = self.interm_target_vrt_nodata_val)
        
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.batch_size, #None, # manual batch of size 1
            num_workers=self.num_workers,
            pin_memory=False,
            # collate_fn = lambda x : x,
            worker_init_fn=ds.seed_worker,
            generator=self.g,
        )
        
        # initialize lists/accumulators
        if self.evaluate:
            # set the cumulative confusion matrix to 0
            self._reset_cm()       
            if seg_criterion is not None:
                seg_losses = [0] * len(dataloader)
                valid_px_list = [0] * len(dataloader)
            if seg_criterion_2 is not None:
                seg_bin_losses = [0] * len(dataloader)
                valid_px_bin_list = [0] * len(dataloader)
            if self.sem_bot:
                pos_error = np.zeros(self.n_interm_targets)
                neg_error = np.zeros(self.n_interm_targets)
                n_pos_pix = np.zeros(self.n_interm_targets)
                n_neg_pix = np.zeros(self.n_interm_targets)
                if detailed_regr_metrics:
                    sse = np.zeros(self.n_interm_targets) # sum of squared errors
                    valid_regr_counts = np.zeros(self.n_interm_targets)
                    sum_regr_targets = np.zeros(self.n_interm_targets)
                    sum_regr_squared_targets = np.zeros(self.n_interm_targets)
                else:
                    sse, valid_regr_counts = None, None
                    sum_regr_targets, sum_regr_squared_targets = None, None
                if regr_criteria is not None:
                    regr_losses = [[0] * len(dataloader) for _ in range(self.n_interm_targets)]
                    regr_weights_list = [[0] * len(dataloader) for _ in range(self.n_interm_targets)]
                if correction_penalizer is not None:
                    corr_losses = [0] * len(dataloader)
                    valid_px_corr_list = [0] * len(dataloader)
                regr_pred_pts = [[] for _ in range(self.n_interm_targets)]
                regr_target_pts = [[] for _ in range(self.n_interm_targets)]
                
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        for batch_idx, data in progress_bar:
            batch_data, (target_data, interm_target_data), margins, input_nodata_mask, batch_template_fn = data
            # template_fn = fns.iloc[0]
            batch_tile_num = [self.exp_utils.tilenum_extractor[0](fn) for fn in batch_template_fn]
            progress_bar.set_postfix_str('Tile(s): {}'.format(batch_tile_num))

            # compute forward pass and aggregate outputs
            outputs, losses, nopred_mask = self._infer_sample(batch_data, margins, 
                                                  seg_criterion=seg_criterion, 
                                                  seg_criterion_2=seg_criterion_2,
                                                  regr_criteria=regr_criteria,
                                                  correction_penalizer=correction_penalizer)
            if self.sem_bot:
                output, rule_output, corr_output, interm_output = outputs
                (seg_loss, valid_px), (seg_bin_loss, valid_bin_px), \
                    (regr_loss, regr_weights), (corr_loss, valid_corr_px) = losses
            else:
                output = outputs
                (seg_loss, valid_px), (seg_bin_loss, valid_bin_px) = losses
            # store validation losses
            if self.evaluate:
                if seg_criterion is not None:
                    seg_losses[batch_idx] = seg_loss
                    valid_px_list[batch_idx] = valid_px
                if seg_criterion_2 is not None:
                    seg_bin_losses[batch_idx] = seg_bin_loss
                    valid_px_bin_list[batch_idx] = valid_bin_px
                if self.sem_bot:
                    if regr_criteria is not None:
                        for i in range(self.n_interm_targets):
                            regr_losses[i][batch_idx] = regr_loss[i]
                            regr_weights_list[i][batch_idx] = regr_weights[i]
                    if correction_penalizer is not None:
                        corr_losses[batch_idx] = corr_loss
                        valid_px_corr_list[batch_idx] = valid_corr_px

            # compute hard predictions and update confusion matrix
            output = output.numpy()
            nodata_mask = np.logical_or(input_nodata_mask.numpy(), nopred_mask.numpy())
            if self.sem_bot:
                rule_output, corr_output= rule_output.numpy(), corr_output.numpy(), 
                interm_output = interm_output.numpy() 
                # postprocess the intermediate regression outputs
                postproc_interm_output = [None] * interm_output.shape[1]
                for i in range(self.n_interm_targets):
                    # scale back to the initial range
                    postproc_interm_output[i] = self.exp_utils.postprocess_regr_predictions(interm_output[:, i], i)
            else:
                rule_output = None
                postproc_interm_output = None
                
            target_data = target_data.numpy()
            if self.sem_bot:
                interm_target_data = [t.numpy() for t in interm_target_data]
            else:
                interm_target_data = None
            output_hard, output_hard_2, output_hard_1, rule_output_hard, _ = self._get_decisions(actv=output, 
                                                                                              target_data=target_data, 
                                                                                              rule_actv=rule_output, 
                                                                                              interm_actv=postproc_interm_output, 
                                                                                              interm_target_data=interm_target_data,
                                                                                              nodata_mask=nodata_mask)
            
            # restore nodata values from inputs + missing predictions
            if np.any(nodata_mask):
                rep_mask = np.repeat(nodata_mask[:, np.newaxis, ...], output.shape[1], axis = 1)
                output[rep_mask] = self.exp_utils.f_out_nodata_val
                if self.sem_bot:
                    rule_output[rep_mask] = self.exp_utils.f_out_nodata_val
                    corr_output[rep_mask] = self.exp_utils.f_out_nodata_val
                    for i in range(self.n_interm_targets):
                        postproc_interm_output[i][nodata_mask] = self.exp_utils.f_out_nodata_val 
            if self.save_error_map: 
                valid_mask = ~nodata_mask
                if self.decision == 'f':
                    main_target = target_data
                    valid_mask *= (main_target != self.target_vrt_nodata_val)
                    seg_error_map = get_seg_error_map(pred=output_hard, 
                                                    target=main_target, 
                                                    valid_mask=valid_mask, 
                                                    n_classes=self.exp_utils.n_classes)
                else:
                    seg_error_map_1 = get_seg_error_map(pred=output_hard_1, 
                                                    target=target_data[:, 0], 
                                                    valid_mask=valid_mask*(target_data[:, 0]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_1)
                    seg_error_map_2 = get_seg_error_map(pred=output_hard_2, 
                                                    target=target_data[:, 1], 
                                                    valid_mask=valid_mask*(target_data[:, 1]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_2)
                    # 0: no error, 1: forest type error, 2: presence of forest error, 3: both errors
                    seg_error_map = (seg_error_map_1>0).astype(np.uint8)
                    seg_error_map[seg_error_map_2>0] += 2
                    
            if self.sem_bot:
                # compute error maps and collect some regression points
                if self.save_error_map or self.evaluate:
                    regr_error_map = [None] * self.n_interm_targets
                    for i in range(self.n_interm_targets):
                        valid_mask = (interm_target_data[i] != self.interm_target_vrt_nodata_val[i]) * ~nodata_mask
                        regr_error_map[i] = get_regr_error_map(pred=postproc_interm_output[i], 
                                                    target=interm_target_data[i], 
                                                    valid_mask=valid_mask)
                        if self.evaluate:
                            if self.decision == 'f':
                                fpa_target_data = target_data # used to indicate presence or absence of forest
                            else:
                                fpa_target_data = target_data[:, -1] 
                            pos_err, neg_err, n_pos, n_neg = get_mae(regr_error_map[i], 
                                                                            fpa_target_data, 
                                                                            valid_mask=valid_mask)
                            n_pos_pix[i] += n_pos; n_neg_pix[i] += n_neg
                            pos_error[i] += pos_err; neg_error[i] += neg_err
                            
                            if detailed_regr_metrics:
                                sse[i] += np.sum(regr_error_map[i][valid_mask]**2)
                                valid_regr_counts[i] += np.sum(valid_mask)
                                sum_regr_targets[i] += np.sum(interm_target_data[i][valid_mask])
                                sum_regr_squared_targets[i] += np.sum(interm_target_data[i][valid_mask]**2)
                            # store some of the regression points for a scatter plot
                            idx = np.unravel_index(np.random.choice(postproc_interm_output[i].size, regr_pts_per_tile), postproc_interm_output[i].shape)
                            mask = valid_mask[idx]
                            regr_pred_pts[i] = np.concatenate((regr_pred_pts[i], postproc_interm_output[i][idx][mask]), axis = 0)
                            regr_target_pts[i] = np.concatenate((regr_target_pts[i], interm_target_data[i][idx][mask]), axis = 0)
                else:
                    regr_error_map = None

            # write outputs 
            for i in range(len(batch_template_fn)):
                tile_num = batch_tile_num[i]
                template_fn = batch_template_fn[i]
                if self.save_hard or self.save_soft or self.save_corr or self.save_interm:   
                    writer = Writer(self.exp_utils, tile_num, template_fn, 
                                    template_scale = self.exp_utils.input_scales[0], 
                                    dest_scale=self.exp_utils.target_scale)
                    # main segmentation output
                    writer.save_seg_result(self.output_dir, 
                                            save_hard = self.save_hard, output_hard = output_hard[i], 
                                            save_soft = self.save_soft, output_soft = output[i], 
                                            colormap = self.exp_utils.colormap)
                    if self.binary_map:
                        # binary forest/non-forest
                        writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = output_hard_2[i], 
                                                save_soft = False, output_soft = None, 
                                                suffix = self.exp_utils.suffix_2, 
                                                colormap = self.exp_utils.colormap_2)
                        
                        if self.decision == 'h':
                            # forest type
                            writer.save_seg_result(self.output_dir, 
                                                    save_hard = self.save_hard, output_hard = output_hard_1[i], 
                                                    save_soft = False, output_soft = None, 
                                                    suffix = self.exp_utils.suffix_1, 
                                                    colormap = self.exp_utils.colormap_1)
                    if self.save_error_map:
                        writer.save_seg_result(self.output_dir, 
                                            save_hard = self.save_hard, output_hard = seg_error_map[i], 
                                            save_soft = False, output_soft = None, 
                                            suffix = '_error',
                                            colormap = None)
                
                    if self.sem_bot:
                        # rule output
                        writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = rule_output_hard[i],
                                                save_soft = self.save_soft, output_soft = rule_output[i],
                                                name_hard = 'rule_predictions', name_soft = 'rule_predictions_soft',
                                                colormap = self.exp_utils.colormap
                                                )
                        # change map (before v.s. after correction)
                        corr_change = output_hard * self.exp_utils.n_classes + rule_output_hard
                        corr_change[output_hard == rule_output_hard] = 0
                        corr_change[nodata_mask] = self.exp_utils.i_out_nodata_val
                        writer.save_seg_result(self.output_dir,
                                            save_hard = self.save_hard, output_hard = corr_change[i],
                                                save_soft = False, output_soft = None,
                                                name_hard = 'corr_change', name_soft = None,
                                                colormap = None
                                                )
                        # regression outputs
                        if self.save_interm:
                            for j in range(self.n_interm_targets):
                                source = self.exp_utils.interm_target_sources[j]
                                writer.save_regr_result(self.output_dir, output = postproc_interm_output[j][i],
                                                        name = 'interm_{}_predictions'.format(source))
                                if self.save_error_map:
                                    writer.save_regr_result(self.output_dir, output = regr_error_map[j][i], 
                                                            name = '{}_error_map'.format(source))
                        if self.save_corr:
                            # correction
                            writer.save_regr_result(self.output_dir, output = corr_output[i], 
                                                    name = 'corr_activations')
                            corr_diff = output - rule_output
                            writer.save_regr_result(self.output_dir, output = corr_diff[i], 
                                                    name = 'corr_diff')
                del output
                del output_hard
                del output_hard_2
                del output_hard_1

        ###### compute metrics ######
        
        if self.evaluate:
            # compute confusion matrix and report
            reports = self._compute_metrics()
            # aggregate losses/errors/samples the validation set
            seg_loss = None if seg_criterion is None else np.average(seg_losses, axis = 0, 
                                                                                weights = valid_px_list)
            seg_bin_loss = None if seg_criterion_2 is None else np.average(seg_bin_losses, axis = 0, 
                                                                                weights = valid_px_bin_list)
            if self.sem_bot:
                regr_loss = None if regr_criteria is None else \
                            [np.average(regr_losses[i], axis = 0, weights = regr_weights_list[i]) 
                                for i in range(self.n_interm_targets)]
                corr_loss = None if correction_penalizer is None else np.average(corr_losses, axis = 0, 
                                                                        weights = valid_px_corr_list)
                n_pos_pix[n_pos_pix == 0] = 1
                n_neg_pix[n_neg_pix == 0] = 1
                pos_mean_regr_error = pos_error / n_pos_pix 
                neg_mean_regr_error = neg_error / n_neg_pix
                mean_regr_error = (pos_error + neg_error) / (n_pos_pix + n_neg_pix)
                    
                if detailed_regr_metrics:
                    rmse = np.sqrt(sse/valid_regr_counts)
                    sstot = sum_regr_squared_targets - sum_regr_targets**2/valid_regr_counts
                    r2 = 1 - sse / sstot
                else:
                    rmse, r2 = None, None

                return self.cum_cms, reports, (seg_loss, seg_bin_loss, regr_loss, corr_loss), \
                    (list(mean_regr_error), list(pos_mean_regr_error), list(neg_mean_regr_error)), \
                    (regr_pred_pts, regr_target_pts), (rmse, r2)
            else:
                return self.cum_cms, reports, (seg_loss, seg_bin_loss)
        else:
            return None
        
    def __del__(self):
        shutil.rmtree(self.tmp_dir)

    

