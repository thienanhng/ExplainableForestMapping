import sys
import os
import argparse
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataset import TrainingDataset, MultiTargetTrainingDataset
from utils import ExpUtils
from models import Unet, RuleUnet, pretrained_models
import utils
from copy import deepcopy
import random

from utils.train_utils import WeightedBCEWithLogitsLoss

########################### PARAMETERS ########################################

def get_parser():
    parser = argparse.ArgumentParser(description='Launch model training')
    parser.add_argument('--input_sources', type=str, nargs='*', default=['SI2017', 'ALTI'],
        choices = ['SI2017', 'ALTI'], \
        help='Source of inputs. Order matters.'\
                'Example: --input_sources SI2017 ALTI')
    parser.add_argument('--interm_target_sources', type=str, nargs='*', default=[],
        choices = ['TH', 'TCD1'], \
        help='Sources of supervision for intermediate regression tasks. TH: tree height from the VHM NFI (Vegetation '
        'Height Model, Swiss National Forest Inventory). TCD1: Tree Canopy Density obtained by thresholding the VHM NFI'
        ' at 1m and spatial averaging.')
    parser.add_argument('--data_dir', type=str, help='Directory where the data is stored')
    parser.add_argument('--train_csv_fn', type=str, help='Csv file listing the input and target files to use for training')
    parser.add_argument('--val_csv_fn', type=str, help='Csv file listing the input and target files to use for validation')
    parser.add_argument('--output_dir', type = str, help='Directory where to store models and metrics. '
                        'The name of the directory will be used to name the model and metrics files. '
                        'A "training/" subdirectory will be automatically created)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size used for training')
    parser.add_argument('--patch_size', type=int, default=128, help='Patch size the coarsest modality')
    parser.add_argument('--inference_batch_size', type=int, default=4, help='Batch size used at validation (number of '
                        'tiles in each batch')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train for')
    parser.add_argument('--lr', type=float, nargs='*', default=[1e-5], help='Learning rate. Can be scheduled by '
                        'specifying several values and using "learning_schedule"')
    parser.add_argument('--learning_schedule', type=int, nargs='*', default = [], help='Number of epochs for'
                        'which the learning rate will be set to the corresponding value in "lr". The remaining epochs ' 
                        'are trained with the last value in "lr". "learning_schedule" should have the same number of elements '
                        'as "lr" if "lr" has more than 1 value.')
    parser.add_argument('--n_negative_samples', type=int, nargs='*', default = [], help='Number of negative examples '
                        '(i.e. tiles without forest) to be used for training for each --negative_sampling_schedule period')
    parser.add_argument('--negative_sampling_schedule', type=int, nargs='*', default = [], help='Number of epochs for'
                        'which the number of negative samples will be set to the corresponding value in '
                        '"n_negative_samples". The remaining epochs are trained with all samples.'
                        '"negative_sampling_schedule" should have the same number of elements as "n_negative_samples".')
    parser.add_argument('--decision', type=str, default='f', choices=['f', 'h'], help='Configuration of the segmentation'
                        'task. "f": flat, i.e. all classes at the same level. "h":hierarchical, with "forest type" and'
                        '"forest presence/absence" subtasks')
    parser.add_argument('--weight_bin_loss', action='store_true', help='Whether to weight the binary "forest '
                        'presence/absence" loss pixel-wise using weights corresponding to the inverse of the class '
                        'frequencies in the full set of classes.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to be used by cuda for data '
                        'loading')
    parser.add_argument('--skip_validation', action = 'store_true', help='Whether to skip validation at each epoch or '
                        'not')
    parser.add_argument('--undersample_validation', type=int, default=1, help='If n is the specified value, '
                        'a 1/n random subset of the validation set will be used for validation '
                        'at each epoch (speeds up validation). The subset is drawn at random at each epoch, '
                        'it will thus be different for each epoch')
    parser.add_argument('--padding', type=int, default=64, help='Margins to leave out at validation')
    parser.add_argument('--resume_training', action='store_true', help='Flag to indicate that we want to resume '
                        'training from a pretrained model stored in output_dir/training')
    parser.add_argument('--starting_point', type=str, default=None, help='Name of the starting point to finetune.')
    parser.add_argument('--no_user_input', action='store_true', help='Flag to disable asking user confirmation for '
                        'overwriting files')
    parser.add_argument('--debug', action='store_true', help='Uses a small subset of the training and validation sets'
                        'to accelerate debugging')
    parser.add_argument('--adapt_loss_weights', action='store_true', help='Flag to indicate whether to adapt loss '
                        'weights of the non-binary tasks before each epoch to the ratio of tiles with and without forest')
    parser.add_argument('--regression_loss', type=str, nargs=2, default=['MSElog', 'MSE'], 
                        choices=['MSE', 'MAE', 'MSElog', 'RMSE', 'RMSElog'], help='type of regression loss for each '
                        'intermediate concept')
    parser.add_argument('--penalize_correction', action='store_true', help='Flag to use a L-1 sparsity penalty on the '
                        'correction activations')
    parser.add_argument('--lambda_bin', type=float, default=1.0, help='for the hierarchical decision setting, weight of'
                        'binary segmentation loss (the weight for the "forest type" loss is always set as 1)')
    parser.add_argument('--lambda_sem', type=float, nargs='*', default=[1.], help='common weight of the intermediate '
                        'regression losses of the semantic bottleneck. Values should be between 0 and 1. The other loss '
                        'terms will be weighted by (1 - lambda_sem). "lambda_sem" and "learning_schedule" should have'
                        ' the same number of elements')
    parser.add_argument('--lambda_corr', type=float, default=1., help='weight of the correction sparsity '
                        'penalty')
    parser.add_argument('--epsilon_rule', type=float, default=1e-3, help='stabilizing factor of rule-based log-probabilities')
    parser.add_argument('--random_seed', type=int, default=0, help='Random seed for random, numpy.random and pytorch.')
    
    return parser


class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self):
        self.debug = True
        self.data_dir = '../Data'
        self.input_sources = ['SI2017', 'ALTI'] # ['SI2017'] #
        self.interm_target_sources = ['TH', 'TCD1'] #[] # 
        self.train_csv_fn = 'data/csv/{}_TLM5c_train_with_counts.csv'.format('_'.join(self.input_sources + self.interm_target_sources))
        self.val_csv_fn = 'data/csv/{}_TLM5c_val.csv'.format('_'.join(self.input_sources + self.interm_target_sources))
        self.batch_size = 16 
        self.inference_batch_size = 1
        self.patch_size = 128
        self.padding = 64
        self.num_epochs = 20
        self.lr = [1e-5, 1e-6, 1e-6, 1e-7] 
        self.learning_schedule = [5, 5, 5, 5] 
        self.n_negative_samples = [0, 5, 10, 20, 40, 80, 160, 320, 320, 320]  
        self.negative_sampling_schedule = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2] 
        # supervision parameters
        self.adapt_loss_weights = False
        self.regression_loss = ['MSElog', 'MSE']
        self.penalize_correction = True 
        self.decision = 'h' 
        self.weight_bin_loss = False 
        self.lambda_bin = 1.0 
        self.lambda_sem = [1., 0.75, 0.5, 0.25] 
        self.lambda_corr = 1.
        self.epsilon_rule = 0.001
        self.num_workers = 2
        self.skip_validation = False
        self.undersample_validation = 1
        self.resume_training = True
        self.starting_point = 'sb_epoch_4'
        self.output_dir = 'output/debug'
        self.no_user_input = True
        self.random_seed = 0


########################################################################################################################

def train(args):
    
    #################### ARGUMENT CHECKING ####################################

    exp_name = os.path.basename(args.output_dir)        
    log_fn = os.path.join(args.output_dir, 'training','{}_metrics.pt'.format(exp_name))
    model_fn = os.path.join(args.output_dir, 'training', '{}_model.pt'.format(exp_name))
    
    if args.resume_training:
        if args.starting_point is None:
            raise ValueError('"starting_point" must be specified to resume training from pretrained model') 
        else:
            starting_model_fn = 'output/{0}/training/{0}_model.pt'.format(args.starting_point)
            starting_metrics_fn = 'output/{0}/training/{0}_metrics.pt'.format(args.starting_point)
            
    ############ Check the paths ###########
    if os.path.isfile(model_fn):
        if os.path.isfile(log_fn):
            if args.resume_training:
                if os.path.isfile(starting_model_fn) and model_fn == starting_model_fn:
                    print('Resuming the training process, {} will be updated.'.format(model_fn))
                if os.path.isfile(starting_metrics_fn) and log_fn == starting_metrics_fn:
                    print('Resuming the training process, {} will be updated.'.format(log_fn))
            else:
                print('WARNING: Training from scratch, {} and {} will be overwritten'.format(log_fn, model_fn))
                if not args.no_user_input:
                    print('Continue? (yes/no)')
                    while True:
                        proceed = input()
                        if proceed == 'yes':
                            break
                        elif proceed == 'no':
                            return
                        else:
                            print('Please answer by yes or no')
                            continue
        else:
            if not os.path.isdir(os.path.dirname(log_fn)):
                print('Directory {} does not exist, it will be created'.format(os.path.dirname(log_fn)))
                os.makedirs(os.path.dirname(log_fn))
    else:
        if not os.path.isdir(os.path.dirname(model_fn)):
            print('Directory {} does not exist, it will be created'.format(os.path.dirname(model_fn)))
            os.makedirs(os.path.dirname(model_fn))

    ############ Check other args ############

    n_input_sources = len(args.input_sources)
    n_interm_targets = len(args.interm_target_sources)
    
    use_schedule = len(args.learning_schedule) > 1
    if use_schedule:
        if len(args.lr) != len(args.learning_schedule):
            raise ValueError('lr and learning_schedule should have the same number of elements')
        
        
    use_sb = n_interm_targets > 0  # use a semantic bottleneck approach       
    if use_sb:
        if args.epsilon_rule <= 0 or args.epsilon_rule >= 1/3:
            raise ValueError('epsilon_rule should be in the interval (0, 1/3)')
        if use_schedule:
            if len(args.lambda_sem) != len(args.learning_schedule):
                raise ValueError('lambda_sem and learning_schedule should have the same number of elements')
    else:
        args.lambda_corr = 0.
        args.lambda_sem = [0.] 
        args.epsilon_rule = None
        if args.penalize_correction:
            print('WARNING: penalize_correction will be set to False because no intermediate targets are specified')
            args.penalize_correction = False
        args.regression_loss = None

    if len(args.n_negative_samples) != len(args.negative_sampling_schedule):
        raise ValueError('n_negative_samples and negative_sampling_schedule should have the same number of elements')
    control_training_set = len(args.n_negative_samples) > 0

    if args.undersample_validation < 1:
        raise ValueError('undersample_validation factor should be greater than 1')
    if args.debug:
        args.undersample_validation = 20
        print('Debug mode: only 1/{}th of the validation set will be used'.format(args.undersample_validation))
            
    exp_utils = ExpUtils(args.input_sources, 
                         args.interm_target_sources, 
                         decision = args.decision,
                         epsilon_rule=args.epsilon_rule)
    
    # create dictionary used to save args        
    if isinstance(args, DebugArgs):
        args_dict = args.__dict__.copy()
    else:
        args_dict = vars(args).copy()
    # add the intermediate targets in the args (useful for plotting the loss weights)
    try:
        args_dict['interm_concepts'] = exp_utils.interm_concepts
        try:
            args_dict['interm_target_stds'] = exp_utils.interm_target_stds
            try:
                args_dict['thresholds'] = exp_utils.unprocessed_thresholds
            except AttributeError:
                pass
        except AttributeError:
            pass
    except AttributeError:
            pass
    if args.resume_training:
        # check that the previous args match the new ones
        if os.path.isfile(starting_metrics_fn):
            save_dict = torch.load(starting_metrics_fn) 
        else:
            save_dict = pretrained_models.get_metrics(args.starting_point, os.path.dirname(starting_metrics_fn))
        previous_args_dict = save_dict['args']
        if args_dict != previous_args_dict:
            print('WARNING: The args saved in {} do not match the current args. '
            'The current args will be appended to the existing ones:')
            for key in args_dict:
                current_val =  args_dict[key]
                try:
                    # using tuples because some of the args are already lists (makes the code simpler)
                    if isinstance(previous_args_dict[key], tuple): #avoid nested tuples
                        args_dict[key] = (*previous_args_dict[key], current_val)
                        previous_val = previous_args_dict[key][-1]
                    else:
                        args_dict[key] = (previous_args_dict[key],current_val)
                        previous_val = previous_args_dict[key]
                    try:
                        val_change = previous_val != current_val   
                    except ValueError:
                        val_change = any([x != y for x, y in zip(np.ravel(previous_val), np.ravel(current_val))])
                    if val_change:
                        print('\t{}: previous value {}, current value {}'.format(key, previous_val, current_val))        
                except KeyError:
                    pass
        # check the keys
        keys = ['train_losses', 'train_total_losses', 'proportion_negative_samples', 'model_checkpoints', \
                                                                                                'optimizer_checkpoints', 'random_state']
        if use_sb:
            keys.extend(('train_regression_losses', 'train_correction_penalties'))
        if args.decision == 'h':
            keys.append('train_binary_losses')
        if not args.skip_validation:
            keys.extend(('val_reports', 'val_cms', 'val_epochs', 'val_losses', 'val_total_losses'))
            if use_sb:
                keys.extend(('val_regression_error', 'val_pos_regression_error', 'val_neg_regression_error', 
                             'val_regression_losses', 'val_regression_prediction_points', 'val_regression_target_points'))
                if args.penalize_correction:
                    keys.append('val_correction_penalties')
            if args.decision == 'h':
                keys.append('val_binary_losses')
        keys_not_found = list(k for k in keys if k not in save_dict.keys())
        for key in ['model_checkpoints', 'optimizer_checkpoints']:
            if key in keys_not_found:
                save_dict[key] = []
                keys_not_found.remove(key)                
        if len(keys_not_found) > 0:
            raise KeyError('Did not find ({}) entry(ies) in {}'.format(
                            ', '.join(keys_not_found), 
                            starting_metrics_fn))


    if torch.cuda.is_available():
            device = torch.device("cuda")
    else:
        raise RuntimeError("CUDA is not available")

    print(args.__dict__)
    
    seed = args.random_seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    

    ####################### MODEL SETUP #######################################
    
    # Compute the constant loss weights if adapt_loss_weights is set to False
    if not args.adapt_loss_weights or args.decision == 'h': # use class frequencies corresponding to "positive" images only
        if args.decision == 'h':
            weights = torch.FloatTensor(exp_utils.get_weights(exp_utils.class_freq['seg_1'], 1, 0))
        else:
            weights = torch.FloatTensor(exp_utils.get_weights(exp_utils.class_freq['seg'], 1, 0))
        print('Loss weights: {}'.format(weights))
    else:
        weights = None
    # ignore_index to ignore the forest patch / gehoelzflaeche pixels and/or non-forest pixels
    seg_criterion = nn.CrossEntropyLoss(reduction = 'mean', ignore_index=exp_utils.i_nodata_val, weight=weights.to(device))

    if args.decision == 'h':
        if args.weight_bin_loss:
            weights = torch.zeros((256,))
            valid_weights = torch.FloatTensor(exp_utils.get_weights(exp_utils.class_freq['seg'], 1, 0))
            weights[:valid_weights.shape[0]] = valid_weights
            seg_criterion_2 = WeightedBCEWithLogitsLoss(refined_weight=weights.to(device))
            print('Using a binary cross-entropy loss with class-specific weights {}'.format(valid_weights))
        else:
            seg_criterion_2 = nn.BCEWithLogitsLoss() 
        for i in range(n_input_sources):
            if exp_utils.input_nodata_val[i] is not None:
                print('WARNING: {}th input sources has nodata value {}, '
                      'but torch.nn.BCEWithLogitsLoss used for the binary task does '
                      'not handle nodata values'.format(i, exp_utils.input_nodata_val[i]))
    else:
        seg_criterion_2 = None
        
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
          
    if use_sb:
        model = RuleUnet(encoder_depth=4, 
                decoder_channels=decoder_channels,
                in_channels=exp_utils.input_channels[0], 
                interm_channels=exp_utils.interm_channels,
                corr_channels=exp_utils.corr_channels,
                thresholds=exp_utils.thresholds,
                rules=exp_utils.rules,
                act_encoding=exp_utils.act_encoding,
                classes=exp_utils.output_channels,
                upsample=upsample,
                aux_in_channels=aux_in_channels,
                aux_in_position=aux_in_position,
                decision=args.decision)
        buffers = deepcopy(list(model.segmentation_head.buffers()))

        # regression criteria of intermediate concepts
        regr_criteria = [None] * n_interm_targets
        if not args.skip_validation:
            val_regr_criteria = [None] * n_interm_targets 
        for i, l in enumerate(args.regression_loss):
            # define the module used to compute the loss
            if l == 'MAE':
                loss_module = utils.WeightedMAE
            elif l == 'MSE':  
                loss_module = utils.WeightedMSE
            elif l == 'MSElog':
                loss_module = utils.WeightedMSElog
            elif l == 'RMSE':
                loss_module = utils.WeightedRMSE
            elif l == 'RMSElog':
                loss_module = utils.WeightedRMSElog
            else:
                raise ValueError('Regression loss "{}" not recognized'.format(args.regression_loss))
            # instantiate the loss
            regr_criteria[i] = loss_module(ignore_val=exp_utils.f_nodata_val,
                                            return_weights=False)
            val_regr_criteria[i] = loss_module(ignore_val=exp_utils.f_nodata_val,
                                                return_weights=True) 

        correction_penalizer = lambda x : torch.linalg.norm(x.view(-1), ord = 1) / x.nelement() #L-1 penalty

    else:
        model = Unet(encoder_depth=4, 
                    decoder_channels=decoder_channels,
                    in_channels=exp_utils.input_channels[0], 
                    classes=exp_utils.output_channels,
                    upsample=upsample,
                    aux_in_channels=aux_in_channels,
                    aux_in_position=aux_in_position)

        regr_criteria = None
        val_regr_criteria = None
        correction_penalizer = None

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('{} trainable parameters'.format(total_params))
    optimizer = optim.AdamW(model.parameters(), lr=args.lr[0], amsgrad=True)
    print('Initial learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    
    # load checkpoints if resuming training from existing model
    if args.resume_training:
        # load the state dicts (model and optimizer)
        if os.path.isfile(starting_model_fn):
            starting_model= torch.load(starting_model_fn) #torch.load(model_fn)
        else:
            starting_model = pretrained_models.get_model(args.starting_point, os.path.dirname(starting_model_fn))
        model.load_state_dict(starting_model['model'])
        if exp_utils.sem_bot:
            # restore buffer values from before load_state_dict
            model.segmentation_head.load_buffers(buffers, device = device)
        optimizer.load_state_dict(starting_model['optimizer'])
        for el in optimizer.param_groups:
            el['lr'] = args.lr[0]
        # set the starting epoch
        starting_epoch = starting_model['epoch'] + 1
        # set the random state of when the pretraining was stopped
        random.setstate(starting_model['random_state']['random'])
        np.random.set_state(starting_model['random_state']['numpy'])
        torch.set_rng_state(starting_model['random_state']['pytorch'])
    else:
        save_dict = {
                'args': args_dict,
                'train_losses': [],
                'train_total_losses': [],
                'model_checkpoints': [],
                'optimizer_checkpoints' : [],
                'proportion_negative_samples' : [],
                'random_state': [],
            }
        if use_sb:
            save_dict['train_regression_losses'] = []
            save_dict['train_correction_penalties'] = []
        if args.decision == 'h':
            save_dict['train_binary_losses'] = []
        if not args.skip_validation:
            save_dict['val_reports'] = []
            save_dict['val_cms'] = []
            save_dict['val_epochs'] = []
            save_dict['val_losses'] = []
            save_dict['val_total_losses'] = []
            if use_sb:
                save_dict['val_regression_error'] = []
                save_dict['val_pos_regression_error'] = []
                save_dict['val_neg_regression_error'] = []
                save_dict['val_regression_losses'] = []
                save_dict['val_regression_prediction_points'] = []
                save_dict['val_regression_target_points'] = []
                if args.penalize_correction:
                    save_dict['val_correction_penalties'] = []
            if args.decision == 'h':
                save_dict['val_binary_losses'] = []
        

        starting_epoch = 0
    
    ######################## DATA SETUP #######################################

    # create dataset
    print('Creating dataset...')
    tic = time.time()
    if use_sb:
        dataset = MultiTargetTrainingDataset( 
                    dataset_csv=args.train_csv_fn,
                    n_input_sources=n_input_sources,
                    n_interm_target_sources=n_interm_targets,
                    exp_utils = exp_utils,
                    data_dir=args.data_dir,
                    control_training_set=control_training_set,
                    n_neg_samples = None,
                    patch_size=args.patch_size,
                    verbose=False,
                    debug=args.debug
                    )
    else:
        dataset = TrainingDataset( 
                    dataset_csv=args.train_csv_fn,
                    n_input_sources=n_input_sources,
                    exp_utils = exp_utils,
                    data_dir=args.data_dir,
                    control_training_set=control_training_set,
                    n_neg_samples = None,
                    patch_size=args.patch_size,
                    verbose=False,
                    debug=args.debug
                    )
    print("finished in %0.4f seconds" % (time.time() - tic))
    
    g = torch.Generator()
    if args.resume_training:
        g.set_state(starting_model['random_state']['pytorch_generator'])
    else:
        g.manual_seed(seed)
    

    # create dataloader
    print('Creating dataloader...')
    tic = time.time()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        worker_init_fn=dataset.seed_worker,
        generator=g,
    )
    print("finished in %0.4f seconds" % (time.time() - tic))

    ########################### TRAINING SETUP ################################
    
    # create array containing the number of negatives samples to be selected for each epoch
    n_neg_samples = np.full(args.num_epochs, dataset.n_negatives)
    if control_training_set:
        n_controlled_epochs = min(args.num_epochs, np.sum(args.negative_sampling_schedule))
        n_neg_samples[:n_controlled_epochs] = np.repeat(
                                                args.n_negative_samples, 
                                                args.negative_sampling_schedule
                                                        )[:n_controlled_epochs]
        # clip the array to the total number of negative samples in the dataset
        n_neg_samples[:n_controlled_epochs] = np.minimum(n_neg_samples[:n_controlled_epochs], dataset.n_negatives)

    # create arrays containing the learning rate and the loss terms weights for each epoch
    lambda_sem_list = np.full(args.num_epochs, args.lambda_sem[-1])
    if use_schedule:
        n_controlled_epochs = min(args.num_epochs, np.sum(args.learning_schedule))
        lr_list = np.full(args.num_epochs, args.lr[-1])
        lr_list[:n_controlled_epochs] = np.repeat(args.lr, args.learning_schedule)[:n_controlled_epochs]
        if use_sb:
            lambda_sem_list[:n_controlled_epochs] = np.repeat(args.lambda_sem, args.learning_schedule)[:n_controlled_epochs]
                
    
    ######################### VALIDATION SETUP ################################
    
    if not args.skip_validation:
        # we do not specify a random seed here (we only want the training to be reproducible)
        inference = utils.Inference(model, args.data_dir,
                args.val_csv_fn, exp_utils, output_dir=None, 
                evaluate=True, save_hard=False, save_soft=False, save_error_map=False,
                batch_size=args.inference_batch_size, padding=args.padding, 
                tile_margin=args.padding,
                num_workers=args.num_workers, device=device,
                undersample=args.undersample_validation, decision=args.decision,
                weight_bin_loss=args.weight_bin_loss)
        
    ######################### TRAINING ########################################
    print('Starting training') 
    n_batches_per_epoch = int(len(dataset.fns) * exp_utils.num_patches_per_tile / args.batch_size)

    for i, epoch in enumerate(range(starting_epoch, starting_epoch + args.num_epochs)):
        print('\nTraining epoch: {}'.format(epoch))
        if control_training_set:
            # update the dataset to select the right number of random negative samples
            dataset.select_negatives(n_neg_samples[i])     
            if n_neg_samples[i] != n_neg_samples[i-1] or i==0:
                # recalculate the number of batches per epoch (for the progress bar)
                n_batches_per_epoch = int(len(dataset.fns) * exp_utils.num_patches_per_tile / args.batch_size) 
                if args.adapt_loss_weights:
                    # adapt the loss weights to the new negatives/positives ratio
                    if args.decision == 'f': # the class frequencies are unchanged if args.decision=='h'
                        weights = torch.FloatTensor(
                                    exp_utils.get_weights(exp_utils.class_freq['seg'], dataset.n_positives, n_neg_samples[i])
                                    )
                        print('Updated loss weights: {}'.format(weights))
                        seg_criterion.weight = weights.to(device) 

                    
        # set learning rate            
        if use_schedule:
            if lr_list[i] != lr_list[i-1] and i > 0:
                print('Updated learning rate: {}'.format(lr_list[i]))
                for el in optimizer.param_groups:
                    el['lr'] = lr_list[i]
                    
        if use_sb:         
            print('Lambda_corr: {}, lambda_sem: {}'.format(args.lambda_corr, lambda_sem_list[i]))
        
        # shuffle data at every epoch (placed here so that all the workers use the same permutation)
        dataset.shuffle()

        # forward and backward pass
        regr_only = lambda_sem_list[i]==1.
        training_loss = utils.fit(
                model = model,
                device = device,
                dataloader = dataloader,
                optimizer = optimizer,
                n_batches = n_batches_per_epoch,
                seg_criterion = seg_criterion,
                seg_criterion_2 = seg_criterion_2,
                regr_criteria = regr_criteria,
                correction_penalizer = correction_penalizer,
                lambda_bin = args.lambda_bin,
                lambda_sem = lambda_sem_list[i], 
                lambda_corr = args.lambda_corr,
                regr_only=regr_only,
                weight_bin_loss=args.weight_bin_loss
            )

        # evaluation (validation) 
        if not args.skip_validation: 
            print('Validation')
            results = inference.infer(seg_criterion=None if regr_only else seg_criterion, 
                                        seg_criterion_2=None if regr_only else seg_criterion_2, 
                                        regr_criteria= val_regr_criteria, 
                                        correction_penalizer=None if regr_only else correction_penalizer)
            if use_sb:
                cm, report, val_losses, (val_regr_error, val_pos_regr_error, val_neg_regr_error), (regr_pred_pts, regr_target_pts), _ = results
            else:
                cm, report, val_losses = results
            # collect individual validation losses and compute total validation loss
            val_loss, val_loss_2, *other_losses = val_losses
            val_total_loss = 0 if regr_only else (1-lambda_sem_list[i]) * val_loss
            if args.decision == 'h' and not regr_only:
                val_total_loss += (1-lambda_sem_list[i]) * args.lambda_bin * val_loss_2
            if use_sb:
                val_regr_losses, val_correction_penalty = other_losses
                val_total_loss += sum(val_regr_losses) * lambda_sem_list[i]
                if args.penalize_correction and not regr_only:
                    val_total_loss += (1-lambda_sem_list[i]) * args.lambda_corr * val_correction_penalty


        # update and save dictionary containing metrics and checkpoints
        
        random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.get_rng_state()
        torch_generator_random_state = g.get_state()
        random_state_dict = {'random': random_state,
                                     'numpy': np_random_state,
                                     'pytorch': torch_random_state,
                                     'pytorch_generator': torch_generator_random_state}
        
        save_dict['proportion_negative_samples'].append(n_neg_samples[i]/dataset.n_fns_all)
        save_dict['model_checkpoints'].append(deepcopy(model.state_dict()))
        save_dict['optimizer_checkpoints'].append(deepcopy(optimizer.state_dict()))
        save_dict['args']['num_epochs'] = epoch + 1 # number of epochs already computed
        save_dict['random_state'].append(random_state_dict)

        # store training losses
        train_loss, train_binary_loss, train_regr_loss, train_correction_penalty = training_loss
        training_total_loss = 0 if regr_only else (1-lambda_sem_list[i]) * train_loss 
        if args.decision == 'h' and not regr_only:
                training_total_loss += (1-lambda_sem_list[i]) * args.lambda_bin * train_binary_loss 
        if use_sb:
            training_total_loss += sum(train_regr_loss) * lambda_sem_list[i]
            if args.penalize_correction and not regr_only:
                training_total_loss += (1-lambda_sem_list[i]) * args.lambda_corr * train_correction_penalty 
        save_dict['train_total_losses'].append(training_total_loss)        
        save_dict['train_losses'].append(train_loss)
        if use_sb:
            save_dict['train_regression_losses'].append(train_regr_loss)
            save_dict['train_correction_penalties'].append(train_correction_penalty)
        if args.decision == 'h':
            save_dict['train_binary_losses'].append(train_binary_loss)
        
        # store validation losses/metrics
        if not args.skip_validation: 
            save_dict['val_reports'].append(report)
            save_dict['val_cms'].append(deepcopy(cm)) # deepcopy is necessary
            save_dict['val_epochs'].append(epoch)
            save_dict['val_total_losses'].append(val_total_loss)
            save_dict['val_losses'].append(val_loss)
            if use_sb:
                save_dict['val_regression_error'].append(val_regr_error)
                save_dict['val_pos_regression_error'].append(val_pos_regr_error)
                save_dict['val_neg_regression_error'].append(val_neg_regr_error)
                save_dict['val_regression_losses'].append(val_regr_losses)
                save_dict['val_regression_prediction_points'].append(regr_pred_pts)
                save_dict['val_regression_target_points'].append(regr_target_pts)
                if args.penalize_correction:
                    save_dict['val_correction_penalties'].append(val_correction_penalty)
            if args.decision == 'h':
                save_dict['val_binary_losses'].append(val_loss_2)
                
        with open(log_fn, 'wb') as f:
            torch.save(save_dict, f)

        # save last checkpoint in a separate file
        last_checkpoint = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch,
                            'model_params': {'decision': args.decision,
                                             'epsilon_rule': args.epsilon_rule},
                            'random_state': random_state_dict} 
        torch.save(last_checkpoint, model_fn)

########################################################################################################################

if __name__ == "__main__":
    if len(sys.argv) < 2:
        args = DebugArgs() # enables to run the script without arguments
    else:
        parser = get_parser()
        args = parser.parse_args()

    train(args)