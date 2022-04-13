import os
import rasterio
import shutil
import random
from osgeo import gdal
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from .eval_utils import cm2rates, rates2metrics, my_confusion_matrix, get_regr_error_map, get_regr_error, get_seg_error_map
from dataset import InferenceDataset
from utils.ExpUtils import I_NODATA_VAL, F_NODATA_VAL
from .write_utils import Writer
import random
from tqdm import tqdm
import gc


class Inference():
    """
    Class to perform inference and evaluate predictions on a set of samples. If used for validation during training, 
    the class must be instantiated once before training and Inference.infer() can be called at each epoch.
    Virtual mosaics with all the tiles for each source are created so that the Dataset can sample patches that overlap 
    several neighboring tiles. If they exist, the nodata values of the inputs rasters are used to fill the gaps, 
    otherwise a new nodata value is introduced depending on the raster data type. When calling infer(), the criteria 
    ignore_index attributes are modified accordingly.
    """
    def __init__(self, model, file_list, exp_utils, output_dir = None, 
                        evaluate = True, save_hard = True, save_soft = True, save_error_map = False,
                        batch_size = 32, num_workers = 0, device = 0, undersample = 1, decision = 'f'):

        """
        Args:
            - model (nn.Module): model to perform inference with
            - file_list (str): csv file containing the files to perform inference on (1 sample per row)
            - exp_utils (ExpUtils): object containing information of the experiment/dataset
            - output_dir (str): directory where to write output files
            - evaluate (bool): whether to evaluate the predictions
            - save_hard (bool): whether to write hard predictions into image files
            - save_soft (bool): whether to write soft predictions into image files
            - batch_size (int): batch size
            - num_workers (int): number of workers to use for the DataLoader. Recommended value is 0 because the tiles
                are processed one by one
            - device (torch.device): device to use to perform inference 
            - undersample (int): undersampling factor to reduction the size of the dataset. Example: if undersample = 100, 
                1/100th of the dataset's samples will be randomly picked to perform inference on.
        """

        self.evaluate = evaluate  
        self.save_hard = save_hard
        self.save_soft = save_soft
        self.model = model
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.undersample = undersample
        self.exp_utils = exp_utils
        self.binary_map = self.exp_utils.decision_func_2 is not None # generate binary map
        self.patch_size = self.exp_utils.patch_size
        self.decision = decision
        self.input_vrt_fn = None # used to indicate that virtual raster mosaic(s) has not been created yet
        self.save_error_map = save_error_map

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

        # Define predictions averaging kernel
        self.kernel = torch.from_numpy(self.exp_utils.get_inference_kernel()) #.to(device)

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
            self.input_col_names = 'input'
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
        # if self.evaluate:
        #     self.cum_cm.fill(0)
        #     if self.binary_map:
        #         self.cum_cm_2.fill(0)
        #     if self.decision == 'h':
        #         self.cum_cm_1.fill(0)
        #     if self.sem_bot:
        #         self.cum_cm_rule.fill(0)
        if self.evaluate:
            for key in self.cum_cms:
                self.cum_cms[key].fill(0)
                

    def _get_decisions(self, actv, target_data, rule_actv=None, interm_actv=None, interm_target_data=None):
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
            if self.evaluate: 
                target = target_data
                if self.binary_map:
                    # get naive binary output
                    target_2 = self.exp_utils.target_recombination(target_data)
                    #if self.exp_utils.decision_func_2 is not None:
                    output_hard_2 = self.exp_utils.decision_func_2(output_2)
                    if self.sem_bot:
                        rule_target = target_data
                        output_hard_rule_2 = self.exp_utils.rule_decision_func_2(output_rule_2)
         
        else:
            # define the outputs 
            output_1 = actv[:-1]
            output_2 = actv[-1]
            if rule_actv is not None:
                output_rule_1, output_rule_2 = rule_actv[:-1], rule_actv[-1]
            # compute hard predictions
            output_hard_1 = self.exp_utils.decision_func(output_1) # ForestType
            output_hard_2 = self.exp_utils.decision_func_2(output_2) # forest presence/absence
            output_hard = (output_hard_1 + 1) * output_hard_2 # apply decision tree -> TLM4c
            if rule_actv is not None:
                output_hard_rule_1 = self.exp_utils.rule_decision_func(output_rule_1)
                output_hard_rule_2 = self.exp_utils.rule_decision_func_2(output_rule_2)
                output_hard_rule = (output_hard_rule_1 + 1) * output_hard_rule_2
            else:
                output_hard_rule = None
            # define the targets
            if self.evaluate or self.save_error_map:
                target = target_data[-1] # TLM4c
                target_1 = target_data[0] # ForestType
                if output_hard_2 is not None:
                    target_2 = target_data[1] # forest presence/absence
                if output_hard_rule is not None:
                    rule_target = target_data[-1] # TLM4c
        if self.sem_bot:
            output_hard_regr = [None] * len(interm_actv)
            for i, t in enumerate(self.exp_utils.unprocessed_thresholds):
                rep_thresh = np.tile(t[:, np.newaxis, np.newaxis], (1, *interm_actv[i].shape))
                output_hard_regr[i] = np.sum(interm_actv[i] > rep_thresh, axis = 0) 
        else:
            output_hard_regr = None
                    
                    
                
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
                                                                            target_cat, #.numpy()),
                                                                            np.ravel(output_hard_regr[i])[mask], 
                                                                            len(self.exp_utils.thresholds[i]) + 1)

        return output_hard, output_hard_2, output_hard_1, output_hard_rule, output_hard_regr

    def _compute_metrics(self):
        """Compute classification metrics from confusion matrices"""
        reports = {}
        for key in self.cum_cms:
            reports[key] = rates2metrics(cm2rates(self.cum_cms[key]), self.exp_utils.class_names[key])
        return reports

    def _infer_sample(self, data, coords, dims, margins, 
                      seg_criterion = None, seg_criterion_2 = None,
                      regr_criteria = None, correction_penalizer = None):
        """Performs inference on one (multi-source) input accessed through dataset ds, with multiple outputs."""

        # compute dimensions of the output
        s = self.exp_utils.target_scale
        height, width = (d * s for d in dims)
        top_margin, left_margin, bottom_margin, right_margin = (m*s for m in margins)

        # initialize accumulators
        output = torch.zeros((self.exp_utils.output_channels, height, width), dtype=torch.float32)
        counts = torch.zeros((height, width), dtype=torch.float32)

        if self.sem_bot:
            # this assumes the the intermediate predictions, correction and the rule-based predictions has the same resolution 
            # than the main target, and that the (complete) correction output has the same depth as the main output
            interm_output = [None] * self.n_interm_targets
            for i in range(self.n_interm_targets):
                interm_output[i] = torch.zeros((self.exp_utils.interm_channels[i],height, width), dtype=torch.float32)
            corr_output = torch.zeros((self.exp_utils.output_channels,height, width), dtype=torch.float32) 
            rule_output = torch.zeros((self.exp_utils.output_channels, height, width), dtype=torch.float32)

        inputs, targets, interm_targets = data
        num_batches = len(inputs[0])
        if self.evaluate:
            if seg_criterion is not None:
                seg_losses = [0] * num_batches
                valid_px_list = [0] * num_batches
            if seg_criterion_2 is not None:
                seg_bin_losses = [0] * num_batches
                valid_px_bin_list = [0] * num_batches
            if self.sem_bot:
                if regr_criteria is not None:
                    regr_losses = [[0] * num_batches for _ in range(self.n_interm_targets)]
                    regr_weights_list = [[0] * num_batches for _ in range(self.n_interm_targets)]
                if correction_penalizer is not None:
                    corr_losses = [0] * num_batches
                    valid_px_corr_list = [0] * num_batches
        # iterate over batches of small patches
        for batch_idx in range(num_batches):
            # get the prediction for the batch
            input_data = [data[batch_idx].to(self.device) for data in inputs]
            if targets is not None:
                target_data = targets[batch_idx].to(self.device) 
            if interm_targets is not None:
                interm_target_data = [data[batch_idx].to(self.device) for data in interm_targets]
            with torch.no_grad():
                # forward pass
                if self.sem_bot:
                    t_main_actv, t_rule_categories, t_corr_actv, t_interm_actv = self.model(*input_data)
                else:
                    t_main_actv = self.model(*input_data)
                # compute validation losses
                if self.evaluate:
                    if seg_criterion is not None:
                        if seg_criterion_2 is not None:
                            seg_actv, bin_seg_actv = t_main_actv[:, :-1], t_main_actv[:, -1]
                            seg_target, bin_seg_target = target_data[:, 0], target_data[:, 1].float() # BCE loss needs float
                            # compute validation loss for binary subtask (last two channels)
                            bin_seg_mask = bin_seg_target != self.target_vrt_nodata_val # custom ignore_index
                            seg_bin_losses[batch_idx] = seg_criterion_2(bin_seg_actv[bin_seg_mask], bin_seg_target[bin_seg_mask]).item()
                            valid_px_bin_list[batch_idx] = torch.sum(bin_seg_mask).item()
                        else:
                            seg_actv = t_main_actv
                            seg_target = target_data #.squeeze(1)
                        # main loss
                        
                        seg_mask = seg_target != seg_criterion.ignore_index
                        seg_losses[batch_idx] = seg_criterion(seg_actv, seg_target).item()
                        valid_px_list[batch_idx] = torch.sum(seg_mask).item()
                    if self.sem_bot:
                        if regr_criteria is not None:
                            for i in range(len(regr_criteria)):
                                rl, rw = regr_criteria[i](t_interm_actv[:,i], 
                                                            interm_target_data[i])
                                regr_losses[i][batch_idx], regr_weights_list[i][batch_idx] = rl.item(), rw.item()
                        if correction_penalizer is not None:
                            # potentially biased estimation because of the margins
                            corr_losses[batch_idx] = correction_penalizer(t_corr_actv).item()
                            valid_px_corr_list[batch_idx] = t_corr_actv.shape[0] 
                        
                # move predictions to cpu
                main_pred = self.seg_normalization(t_main_actv).cpu()
                if self.sem_bot:
                    rule_pred = self.exp_utils.prob_encoding[t_rule_categories.cpu()].movedim((0, 3, 1, 2), (0, 1, 2, 3))
                    corr_actv = t_corr_actv.cpu()
                    interm_pred = t_interm_actv.cpu() 
            # accumulate the batch predictions
            for j in range(main_pred.shape[0]):
                x, y =  coords[batch_idx][j]
                x_start, x_stop = x*s, (x+self.patch_size)*s
                y_start, y_stop = y*s, (y+self.patch_size)*s
                counts[x_start:x_stop, y_start:y_stop] += self.kernel
                output[:, x_start:x_stop, y_start:y_stop] += main_pred[j] * self.kernel
                if self.sem_bot:
                    corr_output[:, x_start:x_stop, y_start:y_stop] += corr_actv[j] * self.kernel
                    rule_output[:, x_start:x_stop, y_start:y_stop] += rule_pred[j] * self.kernel
                    # assumes the intermediate predictions are at the same resolution than the main predictions
                    for i in range(self.n_interm_targets):
                        interm_output[i][:, x_start:x_stop, y_start:y_stop] += interm_pred[j, i] * self.kernel
                
        # normalize the accumulated predictions
        counts = torch.unsqueeze(counts, dim = 0)
        mask = counts != 0

        rep_mask = mask.expand(output.shape[0], -1, -1)
        rep_counts = counts.expand(output.shape[0], -1, -1)
        output[rep_mask] = output[rep_mask] / rep_counts[rep_mask]
        if self.sem_bot:
            rule_output[rep_mask] = rule_output[rep_mask] / rep_counts[rep_mask]
            corr_output[rep_mask] = corr_output[rep_mask] / rep_counts[rep_mask]
            for i in range(self.n_interm_targets):
                rep_mask = mask.expand(interm_output[i].shape[0], -1, -1)
                rep_counts = counts.expand(interm_output[i].shape[0], -1, -1)
                interm_output[i][rep_mask] = interm_output[i][rep_mask] / rep_counts[rep_mask]
        
        # aggregate losses
        if self.evaluate:
            if seg_criterion is None:
                seg_loss, total_valid_px = None, None
            else:
                seg_loss, total_valid_px = self._aggregate_batch_losses(seg_losses, 
                                                                        valid_px_list)
            if seg_criterion_2 is None:
                seg_bin_loss, total_valid_bin_px = None, None
            else:
                seg_bin_loss, total_valid_bin_px = self._aggregate_batch_losses(seg_bin_losses, 
                                                                                valid_px_bin_list)
            if self.sem_bot:                                                                    
                if regr_criteria is None:
                    regr_loss, total_valid_regr_px = None, None
                else:
                    regr_loss = [0]*len(regr_criteria)
                    total_valid_regr_px = [0]*len(regr_criteria)
                    for i in range(len(regr_criteria)):
                        regr_loss[i], total_valid_regr_px[i] = self._aggregate_batch_losses(regr_losses[i], 
                                                                                        regr_weights_list[i])
                if correction_penalizer is None:
                    corr_loss, total_valid_corr_px = None, None
                else:
                    corr_loss, total_valid_corr_px = self._aggregate_batch_losses(corr_losses, 
                                                                                valid_px_corr_list)
        else:
            seg_loss, total_valid_px = None, None 
            seg_bin_loss, total_valid_bin_px = None, None
            if self.sem_bot:
                regr_loss, total_valid_regr_px = None, None
                corr_loss, total_valid_corr_px = None, None
        # remove margins
        output = output[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
        if self.sem_bot:
            rule_output = rule_output[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
            corr_output = corr_output[:, top_margin:height-bottom_margin, left_margin:width-right_margin]
            interm_output = [t[:, top_margin:height-bottom_margin, left_margin:width-right_margin] for t in interm_output]
            return (output, rule_output, corr_output, interm_output), \
                    ((seg_loss, total_valid_px), (seg_bin_loss, total_valid_bin_px), \
                        (regr_loss, total_valid_regr_px), (corr_loss, total_valid_corr_px))
        else:
            return output, ((seg_loss, total_valid_px), (seg_bin_loss, total_valid_bin_px))
    
    @staticmethod            
    def _aggregate_batch_losses(loss_list, valid_px_list):
        total_valid_px = sum(valid_px_list)
        if total_valid_px > 0:
            seg_loss = np.average(loss_list, axis = 0, weights = valid_px_list)
        else:
            seg_loss = 0
        return seg_loss, total_valid_px

    def infer(self, seg_criterion = None, seg_criterion_2 = None, regr_criteria = None, correction_penalizer = None, 
                regr_pts_per_tile = 200):
        """
        Perform tile by tile inference on a dataset, evaluate and save outputs if needed

        Args:
            - criterion (nn.Module): criterion used for training, to be evaluated at validation as well to track 
                    overfitting
        """
        self.model.eval()
        
        if self.undersample > 1 or self.input_vrt_fn is None:
            # select sample to perform inference on
            df = self._select_samples()
            # create virtual mosaics (and set nodata values)
            self._get_vrt_from_df(df)
        # set the cumulative confusion matrix to 0
        if self.evaluate:
            self._reset_cm()       
            if seg_criterion is not None:
                seg_losses = [0] * len(df)
                valid_px_list = [0] * len(df)
            if seg_criterion_2 is not None:
                seg_bin_losses = [0] * len(df)
                valid_px_bin_list = [0] * len(df)
            if self.sem_bot:
                pos_error = np.zeros(self.n_interm_targets)
                neg_error = np.zeros(self.n_interm_targets)
                n_pos_pix = np.zeros(self.n_interm_targets)
                n_neg_pix = np.zeros(self.n_interm_targets)

                if regr_criteria is not None:
                    regr_losses = [[0] * len(df) for _ in range(self.n_interm_targets)]
                    regr_weights_list = [[0] * len(df) for _ in range(self.n_interm_targets)]
                if correction_penalizer is not None:
                    corr_losses = [0] * len(df)
                    valid_px_corr_list = [0] * len(df)

                regr_pred_pts = [[] for _ in range(self.n_interm_targets)]
                regr_target_pts = [[] for _ in range(self.n_interm_targets)]
                
        #create dataset
        ds = InferenceDataset(self.input_vrt_fns, 
                              exp_utils=self.exp_utils, 
                              batch_size = self.batch_size, 
                              target_vrt_fn = self.target_vrt_fn,
                              interm_target_vrt_fn= self.interm_target_vrt_fns,
                              input_nodata_val = self.input_vrt_nodata_val,
                              target_nodata_val = self.target_vrt_nodata_val,
                              interm_target_nodata_val = self.interm_target_vrt_nodata_val)
        dataloader = torch.utils.data.DataLoader(
            ds,
            batch_size=None, # manual batching to obtain batches with patches from the same image
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn = lambda x : x
        )
        # iterate over dataset (tile by tile) 
        progress_bar = tqdm(zip(dataloader, df.iterrows()), total=len(df))
        for (batch_data, (target_data, interm_target_data), coords, dims, margins, input_nodata_mask), (tile_idx, fns) in progress_bar:
            template_fn = fns.iloc[0]
            tile_num = self.exp_utils.tilenum_extractor[0](template_fn)
            progress_bar.set_postfix_str('Tiles(s): {}'.format(tile_num))

            # compute forward pass and aggregate outputs
            outputs, losses  = self._infer_sample(batch_data, coords, dims, margins, 
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
                    seg_losses[tile_idx] = seg_loss
                    valid_px_list[tile_idx] = valid_px
                if seg_criterion_2 is not None:
                    seg_bin_losses[tile_idx] = seg_bin_loss
                    valid_px_bin_list[tile_idx] = valid_bin_px
                if self.sem_bot:
                    if regr_criteria is not None:
                        for i in range(self.n_interm_targets):
                            regr_losses[i][tile_idx] = regr_loss[i]
                            regr_weights_list[i][tile_idx] = regr_weights[i]
                    if correction_penalizer is not None:
                        corr_losses[tile_idx] = corr_loss
                        valid_px_corr_list[tile_idx] = valid_corr_px

            # compute hard predictions and update confusion matrix
            output = output.numpy()
            if self.sem_bot:
                rule_output, corr_output= rule_output.numpy(), corr_output.numpy(), 
                interm_output = [t.numpy() for t in interm_output]
                
                # postprocess the intermediate regression outputs
                postproc_interm_output = [None] * len(interm_output)
                for i in range(self.n_interm_targets):
                    # scale back to the initial range
                    interm_output[i] = interm_output[i].squeeze(0)
                    postproc_interm_output[i] = self.exp_utils.postprocess_regr_predictions(interm_output[i], i)
                            
            else:
                rule_output = None
                postproc_interm_output = None
            output_hard, output_hard_2, output_hard_1, rule_output_hard, regr_output_hard = self._get_decisions(actv=output, 
                                                                                              target_data=target_data, 
                                                                                              rule_actv=rule_output, 
                                                                                              interm_actv=postproc_interm_output, 
                                                                                              interm_target_data=interm_target_data)
            
            # restore nodata values found in the inputs
            if np.any(input_nodata_mask):
                rep_mask = np.repeat(input_nodata_mask[np.newaxis, :, :], output.shape[0], axis = 0)
                output[rep_mask] = self.exp_utils.f_out_nodata_val
                output_hard[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if output_hard_1 is not None:
                    output_hard_1[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if output_hard_2 is not None:
                    output_hard_2[input_nodata_mask] = self.exp_utils.i_out_nodata_val
                if self.sem_bot:
                    rule_output[rep_mask] = self.exp_utils.f_out_nodata_val
                    corr_output[rep_mask] = self.exp_utils.f_out_nodata_val
                    for i in range(self.n_interm_targets):
                        postproc_interm_output[i][input_nodata_mask] = self.exp_utils.f_out_nodata_val 
                        regr_output_hard[i][input_nodata_mask] = self.exp_utils.i_out_nodata_val
            if self.save_error_map: 
                valid_mask = ~input_nodata_mask
                if self.decision == 'f':
                    main_target = target_data
                    valid_mask *= (main_target != self.target_vrt_nodata_val)# * ~input_nodata_mask
                    seg_error_map = get_seg_error_map(pred=output_hard, 
                                                    target=main_target, 
                                                    valid_mask=valid_mask, 
                                                    n_classes=self.exp_utils.n_classes)
                else:
                    seg_error_map_1 = get_seg_error_map(pred=output_hard_1, 
                                                    target=target_data[0], 
                                                    valid_mask=valid_mask*(target_data[0]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_1)
                    seg_error_map_2 = get_seg_error_map(pred=output_hard_2, 
                                                    target=target_data[1], 
                                                    valid_mask=valid_mask*(target_data[1]!=self.target_vrt_nodata_val), 
                                                    n_classes=self.exp_utils.n_classes_2)
                    # 0: no error, 1: forest type error, 2: presence of forest error, 3: both errors
                    seg_error_map = (seg_error_map_1>0).astype(np.uint8)
                    seg_error_map[seg_error_map_2>0] += 2
                    
            if self.sem_bot:
                # compute error maps and collect some regression points
                if self.save_error_map or self.evaluate:
                    regr_error_map = [None] * self.n_interm_targets
                    for i in range(self.n_interm_targets):
                        valid_mask = (interm_target_data[i] != self.interm_target_vrt_nodata_val[i]) * ~input_nodata_mask
                        regr_error_map[i] = get_regr_error_map(pred=postproc_interm_output[i], 
                                                    target=interm_target_data[i], 
                                                    valid_mask=valid_mask)
                        if self.evaluate:
                            if self.decision == 'f':
                                target_data = target_data 
                            else:
                                target_data = target_data[-1] 
                            pos_err, neg_err, n_pos, n_neg = get_regr_error(regr_error_map[i], 
                                                                            target_data, 
                                                                            interm_target_data[i], 
                                                                            self.interm_target_vrt_nodata_val[i])
                            n_pos_pix[i] += n_pos; n_neg_pix[i] += n_neg
                            pos_error[i] += pos_err; neg_error[i] += neg_err

                            # store some of the regression points for a scatter plot
                            idx = np.unravel_index(np.random.choice(postproc_interm_output[i].size, regr_pts_per_tile), postproc_interm_output[i].shape)
                            mask = valid_mask[idx]
                            regr_pred_pts[i] = np.concatenate((regr_pred_pts[i], postproc_interm_output[i][idx][mask]), axis = 0)
                            regr_target_pts[i] = np.concatenate((regr_target_pts[i], interm_target_data[i][idx][mask]), axis = 0)
                else:
                    regr_error_map = None

            # write outputs 
            if self.save_hard or self.save_soft:   
                writer = Writer(self.exp_utils, tile_num, template_fn, 
                                template_scale = self.exp_utils.input_scales[0], 
                                dest_scale=self.exp_utils.target_scale)
                # main segmentation output
                writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = output_hard, 
                                        save_soft = self.save_soft, output_soft = output, 
                                        colormap = self.exp_utils.colormap)
                if self.binary_map:
                    # binary forest/non-forest
                    writer.save_seg_result(self.output_dir, 
                                            save_hard = self.save_hard, output_hard = output_hard_2, 
                                            save_soft = False, output_soft = None, 
                                            suffix = self.exp_utils.suffix_2, 
                                            colormap = self.exp_utils.colormap_2)
                    
                    if self.decision == 'h':
                        # forest type
                        writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = output_hard_1, 
                                                save_soft = False, output_soft = None, 
                                                suffix = self.exp_utils.suffix_1, 
                                                colormap = self.exp_utils.colormap_1)

                        if self.save_error_map:
                            writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = seg_error_map_1, 
                                                save_soft = False, output_soft = None, 
                                                suffix = '_error_1', 
                                                colormap = None)
                            writer.save_seg_result(self.output_dir, 
                                                save_hard = self.save_hard, output_hard = seg_error_map_2, 
                                                save_soft = False, output_soft = None, 
                                                suffix = '_error_2', 
                                                colormap = None)
                if self.save_error_map:
                    writer.save_seg_result(self.output_dir, 
                                        save_hard = self.save_hard, output_hard = seg_error_map, 
                                        save_soft = False, output_soft = None, 
                                        suffix = '_error',
                                        colormap = None)
            
                if self.sem_bot:
                    # rule output
                    writer.save_seg_result(self.output_dir, 
                                            save_hard = self.save_hard, output_hard = rule_output_hard,
                                            save_soft = self.save_soft, output_soft = rule_output,
                                            name_hard = 'rule_predictions', name_soft = 'rule_predictions_soft',
                                            colormap = self.exp_utils.colormap
                                            )
                    # change map (before v.s. after correction)
                    corr_change = output_hard * self.exp_utils.n_classes + rule_output_hard
                    corr_change[output_hard == rule_output_hard] = 0
                    corr_change[input_nodata_mask] = 0
                    writer.save_seg_result(self.output_dir,
                                           save_hard = self.save_hard, output_hard = corr_change,
                                            save_soft = False, output_soft = None,
                                            name_hard = 'corr_change', name_soft = None,
                                            colormap = None
                                            )
                    # regression outputs
                    for i in range(self.n_interm_targets):
                        source = self.exp_utils.interm_target_sources[i]
                        writer.save_regr_result(self.output_dir, output = postproc_interm_output[i],
                                                name = 'interm_{}_predictions'.format(source))
                        if self.save_error_map:
                            writer.save_regr_result(self.output_dir, output = regr_error_map[i], 
                                                    name = '{}_error_map'.format(source))
                    if self.save_soft:
                        # correction
                        writer.save_regr_result(self.output_dir, output = corr_output, 
                                                name = 'corr_activations')
                        corr_diff = output - rule_output
                        writer.save_regr_result(self.output_dir, output = corr_diff, 
                                                name = 'corr_diff')
            del output
            del output_hard
            del output_hard_2
            del output_hard_1
            gc.collect()

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

                return self.cum_cms, reports, (seg_loss, seg_bin_loss, regr_loss, corr_loss), \
                    (list(mean_regr_error), list(pos_mean_regr_error), list(neg_mean_regr_error)), \
                    (regr_pred_pts, regr_target_pts)
            else:
                return self.cum_cms, reports, (seg_loss, seg_bin_loss)
        else:
            return None
        
    def __del__(self):
        shutil.rmtree(self.tmp_dir)

    

