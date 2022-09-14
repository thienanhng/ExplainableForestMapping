import sys
import os
import argparse
import torch
from models import Unet, RuleUnet
import utils
from utils import ExpUtils
from copy import deepcopy
import random
import numpy as np
from models import pretrained_models

########################### PARAMETERS ########################################

def get_parser():

    parser = argparse.ArgumentParser(description='Launch model inference')
    parser.add_argument('--csv_fn', type=str, help='Path to a CSV file containing at least two columns -- "input" or '
            '"input_x" (x an integer, for multimodal model), "target", and optionally "interm_target_x", that point to '
            'files of the dataset imagery and targets and optionally intermediate concept targets')
    parser.add_argument('--input_sources', type=str, nargs='+', default=['SI2017', 'ALTI'],
        choices = ['SI2017', 'ALTI'],
        help='Source of inputs. Order matters. Example: --input_sources SI2017 ALTI')
    parser.add_argument('--interm_target_sources', type=str, nargs='*', default=[], 
        choices = ['TH', 'TCD1'],
        help='Sources of supervision for intermediate regression tasks. TH: tree height from the VHM NFI (Vegetation '
        'Height Model, Swiss National Forest Inventory). TCD1: Tree Canopy Density obtained by thresholding the VHM NFI'
        ' at 1m and spatial averaging.')
    parser.add_argument('--exp_name', type=str, default='bb', help='Name of the model (training experiment) to use. '
                        'Ignored if model_fn is specified.')
    parser.add_argument('--model_fn', type=str, default=None,
        help='Path to the model file.')
    parser.add_argument('--output_dir', type=str, required = True,
        help='Directory where the output predictions will be stored.')
    parser.add_argument('--overwrite', action="store_true",
        help='Flag for overwriting "output_dir" if that directory already exists.')
    parser.add_argument('--padding', type=int, default=64, help='margin to remove around predictions')
    parser.add_argument('--batch_size', type=int, default=4, help='Number of full tiles in each batch')
    parser.add_argument('--num_workers', type=int, default=0,
        help='Number of workers to use for data loading.')
    parser.add_argument('--save_hard', action="store_true",
        help='Flag that enables saving the "hard" class predictions.')
    parser.add_argument('--save_soft', action="store_true",
        help='Flag that enables saving the "soft" class predictions.')
    parser.add_argument('--save_corr', action="store_true",
        help='Flag that enables saving the correction activations (semantic bottleneck model) as well as an images '
            'indicating class changes before vs. after correction.')
    parser.add_argument('--save_interm', action="store_true",
        help='Flag that enables saving the intermediate predictions (semantic bottleneck model).')
    parser.add_argument('--save_error_map', action="store_true",
        help='Flag that enables saving the error maps for intermediate concept regression predictions. Requires'
            'intermediate targets to be specified in "csv_fn" file')
    parser.add_argument('--evaluate', action='store_true', help='whether to compute metrics on the obtained predictions'
                        '(target must be specified in "csv_fn".')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for random, numpy.random and pytorch.')
    return parser


class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self):
        self.input_sources = ['SI2017', 'ALTI'] #['SI2017'] # 
        self.interm_target_sources = [] # ['TH', 'TCD1'] #
        self.padding = 64
        self.batch_size = 1 # faster with batch size of 1
        self.num_workers = 2
        self.save_hard = False
        self.save_soft = False
        self.save_error_map = False
        self.save_corr = False
        self.save_interm = False
        self.overwrite = True
        set = 'test' 
        self.csv_fn = 'data/csv/{}_TLM5c_{}.csv'.format('_'.join(self.input_sources + self.interm_target_sources), set)
        # self.csv_fn = 'data/csv/{}_{}.csv'.format('_'.join(self.input_sources), set) # for inference without evaluation
        self.exp_name = 'bb' #'bb_wo_alti'
        self.model_fn = None #'output/{0}/training/{0}_model.pt'.format(exp_name)
        self.output_dir = 'output/{}/inference/epoch_19/{}'.format(self.exp_name, set)
        self.random_seed = 0

        self.evaluate = False 


###############################################################################

def infer(args):

    ##################### ARGUMENT CHECKING ###################################

    # check output path
    output_dir = args.output_dir
    if output_dir is None: # defaut directory for output images
        inference_dir = os.path.join(os.path.dirname(os.path.dirname(args.model_fn)), 'inference')
        model_name = os.path.splitext(os.path.basename(args.model_fn))[0]
        output_dir = os.path.join(inference_dir, model_name)
        os.makedirs(output_dir, exist_ok = True)
    else: # custom directory for output images/metrics
        if os.path.exists(output_dir):
            if os.path.isfile(output_dir):
                raise NotADirectoryError("A file was passed as `--output_dir`, please pass a directory!")
            elif len(os.listdir(output_dir)) > 0:
                if args.overwrite:
                    print("WARNING: Output directory {} already exists, we might overwrite data in it!"
                            .format(output_dir))
                else:
                    raise FileExistsError("Output directory {} already exists and isn't empty."
                                            .format(output_dir))
        else:
            print("{} doesn't exist, creating it.".format(output_dir))
            os.makedirs(output_dir)
    if args.evaluate:
            metrics_fn = os.path.join(output_dir, '{}_metrics.pt'.format(exp_name))
            
    # check paths of model and input
    if not os.path.exists(args.csv_fn):
        raise FileNotFoundError("{} does not exist".format(args.csv_fn))
    if args.model_fn is not None: # None means that the model will be downloaded from the web
        if os.path.exists(args.model_fn):
            model_fn = args.model_fn
            exp_name = os.path.basename(os.path.dirname(os.path.dirname(args.model_fn)))
        else:
            raise FileNotFoundError('{} does not exist.'.format(args.model_fn))
             
    seed = args.random_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    ######################### MODEL SETUP #####################################
    
    # check gpu
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    # Set up data and utilities
    if args.model_fn is None:
        model_dir = 'output/{}/training'.format(args.exp_name)
        model_obj = pretrained_models.get_model(args.exp_name, model_dir)
    else:
        model_obj = torch.load(model_fn)
    decision = model_obj['model_params']['decision']
    n_input_sources = len(args.input_sources)
    try:
        epsilon_rule = model_obj['model_params']['epsilon_rule']
    except KeyError:
        epsilon_rule = 1e-3
    exp_utils = ExpUtils(args.input_sources, 
                               args.interm_target_sources, 
                               decision=decision,
                               epsilon_rule=epsilon_rule)
    
    # Set model architecture
    decoder_channels = (256, 128, 64, 32)
    upsample = (True, True, True, False)
    if n_input_sources > 1:
        # 2 input modalities 
        aux_in_channels = exp_utils.input_channels[1] 
        aux_in_position = 1
    else:
        # 1 input modality
        aux_in_channels = None
        aux_in_position = None
    # Create model
    if exp_utils.sem_bot:
        model = RuleUnet(encoder_depth=4, 
                decoder_channels=decoder_channels,
                in_channels = exp_utils.input_channels[0], 
                interm_channels = exp_utils.interm_channels,
                corr_channels = exp_utils.corr_channels,
                thresholds = exp_utils.thresholds,
                rules = exp_utils.rules,
                act_encoding = exp_utils.act_encoding,
                classes = exp_utils.output_channels,
                upsample = upsample,
                aux_in_channels = aux_in_channels,
                aux_in_position = aux_in_position,
                decision=decision)
        buffers = deepcopy(list(model.segmentation_head.buffers()))
    else:
        model = Unet(encoder_depth=4, 
                    decoder_channels=decoder_channels,
                    in_channels = exp_utils.input_channels[0], 
                    classes = exp_utils.output_channels,
                    upsample = upsample,
                    aux_in_channels = aux_in_channels,
                    aux_in_position = aux_in_position)


    model.load_state_dict(model_obj['model'])
    if exp_utils.sem_bot:
        # restore buffer values from before load_state_dict
        model.segmentation_head.load_buffers(buffers, device=device)
    model = model.to(device)

    ####################### INFERENCE #########################################


    inference = utils.Inference(model, args.csv_fn, exp_utils, padding=args.padding, tile_margin=args.padding, 
                                batch_size=args.batch_size, 
                                output_dir=output_dir, evaluate=args.evaluate, 
                                save_hard=args.save_hard, save_soft=args.save_soft, 
                                save_error_map = args.save_error_map, save_corr=args.save_corr, 
                                save_interm=args.save_interm,
                                num_workers=args.num_workers, device=device, decision=decision,
                                random_seed=seed)

    result = inference.infer(detailed_regr_metrics=True)

    ######################### EVALUATION ######################################
    
    if args.evaluate:
        if result is not None:
            cumulative_cm, report, *other_outputs = result
            if isinstance(args, DebugArgs):
                args_dict = args.__dict__
            else:
                args_dict = vars(args).copy()
            if exp_utils.sem_bot:
                args_dict['interm_concepts'] = exp_utils.interm_concepts
            # Save metrics
            d = {
                'args': args_dict,
                'val_reports': report,
                'val_cms': cumulative_cm
            }    
            if exp_utils.sem_bot:
                _, val_regr_error, regr_pts, (rmse, r2) = other_outputs 
                val_regr_error, val_pos_regr_error, val_neg_regr_error = val_regr_error
                d['val_regression_error'] = val_regr_error
                d['val_pos_regression_error'] = val_pos_regr_error
                d['val_neg_regression_error'] = val_neg_regr_error
                d['val_regression_prediction_points'], d['val_regression_target_points'] = regr_pts
                d['val_regression_rmse'] = rmse
                d['val_regression_r2'] = r2
            with open(metrics_fn, 'wb') as f:
                torch.save(d, f)

########################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        args = DebugArgs() # enables to run the script without arguments
    else:
        parser = get_parser()
        args = parser.parse_args()

    infer(args)