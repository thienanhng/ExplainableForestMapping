""" 
Apply the set of rules derived from forest definitions to the intermediate concept targets (TH, TCD), and compare with 
TLM targets.
"""

import sys
import os
import pandas as pd
import numpy as np
import rasterio
import torch
from tqdm import tqdm

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

from utils.ExpUtils import TILENUM_EXTRACTOR
from utils.eval_utils import my_confusion_matrix, rates2metrics, cm2rates

 
############################# CONSTANTS #######################################
COLORMAP =  { 
            0: (0, 0, 0, 0), #non forest 
            1: (21, 180, 0, 255), #open forest
            2: (25, 90, 0, 255), #closed forest
            3: (151, 169, 93, 255), #shrub forest
            4: (252, 255, 51, 255), #non forest or shrub forest
            5: (255, 37, 37, 255), #shrub forest or closed forest
            255: (255, 255, 255, 255) # nodata
            }
n_classes = 4
n_classes_fpa = 2 # forest presence/absence
n_classes_ft = 3 # forest type
nodata_val = 255
undefined_val = 254

############################# PARAMETERS ######################################

csv_fn = os.path.join(project_dir,'data/csv/SI2017_ALTI_TH_TCD1_TLM5c_test.csv') # csv file listing the tiles to process
solve_ambig = True # whether to disambiguate to rule categories containing several forest classes
save_im = True # save the obtained predictions as images
evaluate = True # compute metrics
out_dir = os.path.join(project_dir,'output/rules_on_regr_labels_TH_TCD1/test') # folder where to save the images

############################# PROCESSING ######################################

df = pd.read_csv(csv_fn)
TH_fns = df['interm_target_0']
TCD_fns = df['interm_target_1']
TLM_fns = df['target']
tilenum_extractor = TILENUM_EXTRACTOR['TH']

if not os.path.exists(out_dir):
    os.mkdir(out_dir)
    
cum_cm_main = np.zeros((n_classes, n_classes))
cum_cm_fpa = np.zeros((n_classes_fpa, n_classes_fpa))
cum_cm_ft = np.zeros((n_classes_ft, n_classes_ft))
nodata_val = 255

for i in tqdm(range(len(df))):
    # read data
    with rasterio.open(TH_fns[i], 'r') as f_TH:
        TH_im = f_TH.read(1)
        TH_nodataval = f_TH.profile['nodata']
    with rasterio.open(TCD_fns[i], 'r') as f_TCD:
        TCD_im = f_TCD.read(1)
        TCD_nodataval = f_TCD.profile['nodata']
    with rasterio.open(TLM_fns[i], 'r') as f_TLM:
        TLM_im = f_TLM.read(1)
        profile = f_TLM.profile
        
    im_shape = TH_im.shape
    
    ###### APPLY RULES
    rule_im = np.empty(im_shape, dtype = np.uint8)
    ### non forest (0)
    rule_im[TH_im < 1] = 0 
    rule_im[(TH_im >= 1) * (TH_im < 3) * (TCD_im < 60)] = 0
    rule_im[(TH_im >= 3) * (TCD_im < 20)] = 0
    ### open forest (1)
    rule_im[(TH_im >= 3) * (TCD_im >= 20) * (TCD_im < 60)] = 1
    ### non forest or shrub forest (2)
    rule_im[(TH_im >= 1) * (TH_im < 3) * (TCD_im >= 60)] = 4
    ### shrub forest of closed forest (3)
    rule_im[(TH_im >= 3) * (TCD_im >= 60)] = 5
    
    rule_im[TH_im == TH_nodataval] = nodata_val
    rule_im[TCD_im == TCD_nodataval] = nodata_val
    
    # compute metrics
    if solve_ambig or evaluate:
    
        rule_im = rule_im.flatten()
        rule_im_main = rule_im.copy()
        TLM_im = np.ravel(TLM_im)
        # non forest or shrub forest (4)
        mask_4 = rule_im == 4
        nf_mask_true = TLM_im == 0
        sf_mask_true = TLM_im == 3
        ### disambiguate correct predictions using TLM
        rule_im_main[mask_4 * nf_mask_true] = 0
        rule_im_main[mask_4 * sf_mask_true] = 3
        ### do not disambiguate wrong predictions (assign to undefined_val)
        other_mask_true = 1 - np.logical_or(nf_mask_true, sf_mask_true)
        idx_wrong_4 = np.argwhere(mask_4 * other_mask_true)
        rule_im_main[idx_wrong_4] = undefined_val
        
        # closed forest or shrub forest (5)
        mask_5 = rule_im == 5
        cf_mask_true = TLM_im == 2
        sf_mask_true = TLM_im == 3
        ### disambiguate correct predictions using TLM
        rule_im_main[mask_5 * cf_mask_true] = 2
        rule_im_main[mask_5 * sf_mask_true] = 3
        ### do not disambiguate wrong predictions (assign to undefined_val)
        other_mask_true = 1 - np.logical_or(cf_mask_true, sf_mask_true)
        idx_wrong_5 = np.argwhere(mask_5 * other_mask_true)
        rule_im_main[idx_wrong_5] = undefined_val
        
        nodata_mask = np.logical_or(TLM_im==nodata_val, rule_im_main==nodata_val)
        
        ### forest type (OF, CF, SF)
        rule_im_ft = rule_im_main.copy() - 1
        rule_im_ft[idx_wrong_4] = 2 # assign the NF/SF category to SF forest type
        TLM_im_ft = TLM_im - 1
        ft_nodata = ~((TLM_im > 0) * (TLM_im < 4)) # pixels where the forest type is not known
        error_map_ft = rule_im_ft != TLM_im_ft
        error_map_ft[ft_nodata] = 0 # consider that there is no ft error when ft is unknown in TLM
        
        ### forest presence/absence (NF, F)
        rule_im_fpa = rule_im_main > 0
        rule_im_fpa[idx_wrong_5] = 1 # assign the CF/SF category to presence of forest
        TLM_im_fpa = TLM_im > 0
        error_map_fpa = rule_im_fpa != TLM_im_fpa
        
        if solve_ambig:
            # compute an error map
            # 0: no error, 1: forest type error, 2: presence of forest error, 3: both errors
            error_map = error_map_ft.astype(np.uint8)
            error_map[error_map_fpa] += 2 
            error_map[nodata_mask] = nodata_val
        
        if evaluate:
            ### forest type (OF, CF, SF)
            cum_cm_ft += my_confusion_matrix(TLM_im_ft[~ft_nodata], rule_im_ft[~ft_nodata], n_classes_ft)
            ### main task (NF, OF, CF, SF)
            cum_cm_main += my_confusion_matrix(TLM_im[~nodata_mask], rule_im_main[~nodata_mask], n_classes)
            ### forest presence/absence (NF, F)
            cum_cm_fpa += my_confusion_matrix(TLM_im_fpa[~nodata_mask], rule_im_fpa[~nodata_mask], n_classes_fpa)
            
        
    if save_im:
        tile_num = tilenum_extractor(TH_fns[i])
        if solve_ambig:
            out_fn = os.path.join(out_dir, 'predictions_5c_{}.tif'.format(tile_num))
        else:
            out_fn = os.path.join(out_dir, 'predictions_{}.tif'.format(tile_num))
        with rasterio.open(out_fn, 'w', **profile) as f_out:
            if solve_ambig:
                f_out.write(rule_im_main.reshape(im_shape), 1)
            else:
                f_out.write(rule_im, 1)
            f_out.write_colormap(1, COLORMAP)
        if solve_ambig:
            out_error_fn = os.path.join(out_dir, 'error_{}.tif'.format(tile_num))
            error_profile = profile
            error_profile['colormap'] = None
            with rasterio.open(out_error_fn, 'w', **profile) as f_error_out:
                f_error_out.write(error_map.reshape(im_shape), 1)
                
if evaluate:  
    class_names = (['NF', 'OF', 'CF', 'SF'], ['NF', 'F'], ['OF', 'CF', 'SF'])
    cum_cms = {'main': cum_cm_main, 'pof': cum_cm_fpa, 'ft': cum_cm_ft}
    reports = {}
    for key, cn in zip(cum_cms.keys(), class_names):
        d = cm2rates(cum_cms[key])
        reports[key] = rates2metrics(d, cn)

    save_obj = {'reports': reports,
                'cms': cum_cms}

    torch.save(save_obj, os.path.join(out_dir, 'metrics.pt'))
    
    
    

