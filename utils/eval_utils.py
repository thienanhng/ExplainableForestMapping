"""Functions to compute classification metrics"""

import numpy as np
import torch
from collections import OrderedDict as Dict
from math import sqrt


def my_confusion_matrix(y_true, y_pred, n_targets):
    """adapted from https://stackoverflow.com/questions/59080843/faster-method-of-computing-confusion-matrix"""
    y_true = torch.tensor(y_true, dtype=torch.long).view(-1)
    y_pred = torch.tensor(y_pred, dtype=torch.long).view(-1)
    # treat values outside of [0, n_targets) as pixels to be ignored
    mask_true = (y_true < n_targets) * (y_true >= 0)
    mask_pred = (y_pred < n_targets) * (y_pred >= 0)
    mask = mask_true * mask_pred
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    y = n_targets * y_true + y_pred
    y = torch.bincount(y)
    if len(y) < n_targets * n_targets:
        y = torch.cat((y, torch.zeros(n_targets * n_targets - len(y), dtype=torch.long)))
    y = y.reshape(n_targets, n_targets)
    return y.numpy()

def cm2rates(cm):
    """Extract true positive/negative and false positive/negative rates from a confusion matrix"""
    dict = Dict()
    n_classes = cm.shape[0]
    for c in range(n_classes):
        tp = cm[c,c] # true positives
        idx = [i for i in range(n_classes) if i != c]
        tn = np.sum(cm[idx,idx]) # true negatives
        fp = np.sum(cm[idx,c]) # false positives
        fn = np.sum(cm[c,idx]) # false negatives
        dict[c] = {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    return dict 

def rates2metrics(dict,class_names):
    """
    Computes classification metrics from true positive/negative (tp/tn) and 
    false positive/negative (fp/fn) rates
    Overall metrics are obtained by a weighted average of the class-specific 
    metrics, with weights equal to the class support (number of occurrences)
    Mean metrics are obtained by a non-weighted average of the class-specific 
    metrics

    Args:
        - dict (dictionary): dictionary containing tp, tn, fp, fn values for each class
            (typically obtained with cm2rates())
        - class_names (list of str): class names

    Output:
        - dict_out (dictionary): contains all the computed metrics
    """
    # avoids having to catch zero divisions and 0/0
    div = lambda n, d: n / d if d and n else 0

    dict_out = Dict()
    class_acc = []
    class_prec = []
    class_rec = []
    class_f1 = []
    class_support = []
    # compute support of each class
    for c in dict:
        d = dict[c]
        support = d['tp'] + d['fn']
        class_support.append(support)
    tot_support = sum(class_support)
    # compute metrics for each class
    for c in dict:
        d = dict[c]
        acc = div((d['tp'] + d['tn']) , sum(d.values())) 
        prec = div(d['tp'] , (d['tp'] + d['fp']))
        rec = div(d['tp'] , class_support[c])
        f1 = div(2 * prec * rec , prec + rec)  
        dict_out[class_names[c]] = {'accuracy':acc, 'precision':prec, 
            'recall':rec, 'f1-score':f1, 
            'support (%)':class_support[c]/tot_support*100, 'support':class_support[c]}
        class_acc.append(acc)
        class_prec.append(prec)
        class_rec.append(rec)
        class_f1.append(f1)
    # compute mean metrics
    n_classes = len(class_names)
    dict_out['mean'] = {'accuracy': sum(class_acc)/n_classes,
                        'precision':sum(class_prec)/n_classes, 
                        'recall':sum(class_rec)/n_classes, 
                        'f1-score':sum(class_f1)/n_classes, 
                        'support (%)': 100,
                        'support':sum(class_support)}
    # compute overall metrics                    
    weighted_mean = lambda metric, weight: div(sum([m*w for m,w in zip(metric,weight)]) , sum(weight))
    dict_out['overall'] = {'accuracy': weighted_mean(class_acc, class_support),
                        'precision':weighted_mean(class_prec, class_support), 
                        'recall':weighted_mean(class_rec, class_support), 
                        'f1-score':weighted_mean(class_f1, class_support), 
                        'support (%)': 100,
                        'support':sum(class_support)}
    return dict_out

def get_seg_error_map(pred, target, valid_mask, n_classes):
    error_map = np.zeros_like(target)
    error_map[valid_mask] = target[valid_mask] * n_classes + pred[valid_mask]
    error_map[target == pred] = 0
    return error_map

def get_regr_error_map(pred, target, valid_mask):
    error_map = np.zeros_like(target)
    # print(psutil.virtual_memory()[2])
    error_map[valid_mask] = pred[valid_mask] - target[valid_mask]
    return error_map

def get_mae(error_map, target, valid_mask):
    """Compute the MAE of regression predictions."""
    pos_mask = (target > 0) * valid_mask
    neg_mask = (target == 0) * valid_mask
    pos_mae, neg_mae = np.sum(np.abs(error_map[pos_mask])), np.sum(np.abs(error_map[neg_mask]))
    n_pos, n_neg = np.sum(pos_mask), np.sum(neg_mask)
    return pos_mae, neg_mae, n_pos, n_neg
        
  
    