from abc import abstractmethod, ABC
import numpy as np
from numpy.core.numeric import NaN
from tqdm import tqdm
import torch
import torch.nn as nn
from utils.ExpUtils import F_NODATA_VAL, I_NODATA_VAL


def fit(model, device, dataloader, optimizer, n_batches, seg_criterion, seg_criterion_2, regr_criteria, 
                correction_penalizer = None, lambda_bin = 1.0, lambda_sem = 1.0, lambda_corr = 1.0, regr_only = True):
    """
    Runs 1 training epoch, with intermediate targets.
    n_batches is used to setup the progress bar only. The actual number of batches is defined by the dataloader.
    Returns average losses over the batches. Warning: the average assumes the same number of pixels contributed to the
    loss for each batch (unweighted average over batches)
    """
    model.train()
    regr_losses = []
    if not regr_only:
        losses = []
        binary_losses = []
        correction_penalties = []
    running_loss = 0.0
    dump_period = 100 # period (in number of batches) for printing running loss
    if regr_criteria is not None:
        n_interm_targets = len(regr_criteria)
        
    # train batch by batch
    progress_bar = tqdm(enumerate(dataloader), total=n_batches)
    for batch_idx, data in progress_bar:    
        total_loss = torch.tensor([0.], requires_grad=True, device=device)
        inputs, target, *interm_targets = data
        inputs = [d.to(device) for d in inputs]
        if len(interm_targets) > 0:
            interm_targets = [d.to(device) for d in interm_targets[0]] # need to squeeze interm_target
        target = target.to(device) 

        # collect outputs of forward pass
        optimizer.zero_grad()
        if regr_criteria is None:
            final_actv = model(*inputs)
        else:
            final_actv, _, correction_actv, interm_actv = model(*inputs)
            
        # intermediate concept supervision
        if regr_criteria is not None:
            regr_loss = [0] * n_interm_targets
            for i in range(n_interm_targets):
                regr_loss[i] = regr_criteria[i](
                                    interm_actv[:, i], 
                                    interm_targets[i])
                total_loss = total_loss + lambda_sem * regr_loss[i]
                
        if not regr_only:
            # correction penalty
            if correction_penalizer is not None:
                correction_pen = correction_penalizer(correction_actv) 
                total_loss = total_loss + (1 - lambda_sem) * lambda_corr * correction_pen 
                
            # segmentation loss(es)
            if seg_criterion_2 is not None: # 2 sub-tasks
                seg_actv, bin_seg_actv = final_actv[:, :-1], final_actv[:, -1]
                seg_target, bin_seg_target = target[:, 0], target[:, 1].float()
                # backpropagate for binary subtask (last channels)
                bin_seg_loss = seg_criterion_2(bin_seg_actv, bin_seg_target)
                total_loss = total_loss + (1 - lambda_sem) * lambda_bin * bin_seg_loss 
            else: # only 1 task
                seg_actv = final_actv
                seg_target = target.squeeze(1)
            # main supervision
            seg_loss = seg_criterion(seg_actv, seg_target)
            total_loss = total_loss + (1 - lambda_sem) * seg_loss

        # backward pass
        total_loss.backward()
        optimizer.step()

        # store current losses
        if regr_criteria is not None:
            regr_losses.append([l.item() for l in regr_loss])   
        if not regr_only:
            losses.append(seg_loss.item())
            if seg_criterion_2 is not None:
                binary_losses.append(bin_seg_loss.item())
            if correction_penalizer is not None:
                correction_penalties.append(correction_pen.item())
        
        running_loss += total_loss.item() 
        # print running loss
        if batch_idx % dump_period == dump_period - 1: 
            # this is an approximation because each patch has a different number of valid pixels
            progress_bar.set_postfix(loss=running_loss/dump_period)
            running_loss = 0.0

    # average losses over the epoch
    avg_regr_loss = NaN if regr_criteria is None else np.mean(regr_losses, axis=0)
    avg_loss = NaN if regr_only else np.mean(losses, axis = 0)
    avg_binary_loss = NaN if (seg_criterion_2 is None or regr_only) else np.mean(binary_losses)
    avg_correction_penalty = NaN if (correction_penalizer is None or regr_only) else np.mean(correction_penalties)

    return avg_loss, avg_binary_loss, avg_regr_loss, avg_correction_penalty

def loss_mean(px_error, weights):
    total_weights = torch.sum(weights) #[mask])
    if total_weights > 0:
        l = torch.sum(weights * px_error) / total_weights
    else:
        l = torch.tensor(0.)
    return l, total_weights

def loss_root_mean(px_error, weights, eps = 1e-10):
    """the output total_weights doesnt exactly reflect the actual weighting because of the sqrt"""
    total_weights = torch.sum(weights) #[mask])
    if total_weights > 0:
        total_weights = torch.sqrt(total_weights + eps)
        l = torch.sqrt(torch.sum(weights * px_error) + eps) / total_weights
    else:
        l = torch.tensor(0.)
    return l, total_weights 

class WeightedRegressionLoss(nn.Module, ABC):
    def __init__(self, mean, std, slope=1.0, ignore_val=F_NODATA_VAL, return_weights=False):
        super().__init__()
        self.ignore_val = ignore_val
        self.mean = mean
        self.std = std
        self.slope = slope
        self.return_weights = return_weights
        self.reduction = loss_mean

    def forward(self, pred, target):
        mask = target != self.ignore_val
        px_error = self.get_error(pred[mask], target[mask])
        weights = 1.0 + target[mask] * self.slope 
        l, total_weights = self.reduction(px_error, weights)
        if self.return_weights:
            return l, total_weights
        else:
            return l
        
    @abstractmethod
    def get_error(self, pred, target):
        raise NotImplementedError
    
class WeightedMAE(WeightedRegressionLoss):
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_error(self, pred, target):
        return torch.abs(pred - target)
        
class WeightedMSE(WeightedRegressionLoss):
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def get_error(self, pred, target):
        return (pred - target)**2

class WeightedRMSE(WeightedMSE):
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction = loss_root_mean
    
class WeightedMSElog(WeightedRegressionLoss):
        
    def __init__(self, *args, **kwargs):
        self.eps = 1e-1
        super().__init__(*args, **kwargs)
        
    def get_error(self, pred, target):
        return (torch.log(pred + self.eps) - torch.log(target + self.eps))**2

class WeightedRMSElog(WeightedMSElog):

    def __init__(self, *args, **kwargs):
        self.eps = 1e-3
        super().__init__(*args, **kwargs)
        self.reduction = loss_root_mean
                
class CrossEntropyLossWithCount(nn.CrossEntropyLoss):
    def __init__(self, ignore_index=I_NODATA_VAL, weight=None):
        super().__init__(ignore_index=ignore_index, weight=weight)
        
    @property
    def ignore_val(self):
        return self.ignore_index
    
    @ignore_val.setter
    def ignore_val(self, val):
        self.ignore_index = val
        
    def forward(self, pred, target):
        loss = super().forward(pred, target)
        mask = pred != self.ignore_index
        count = torch.sum(mask.long())
        return loss, count
    
        


