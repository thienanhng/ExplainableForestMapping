import torch
import torch.nn as nn
import torch.nn.functional as f

def complete_flat_correction(corr_activations):
    """add a new channel such that the logits sum to 0 along channels"""
    corr_remainder = -torch.sum(corr_activations, dim=1, keepdim=True)
    return torch.cat((corr_activations, corr_remainder), dim=1)

def complete_hierarchical_correction(corr_activations):
    """add a new channel such that the logits sum to 0 along channels, from the first channel to the penultimate channel
    (last channel is considered as an independend binary sub-task)"""
    corr_remainder = -torch.sum(corr_activations[:, :-1], dim=1, keepdim=True)
    return torch.cat((corr_activations[:, :-1], corr_remainder, corr_activations[:, -1:]), dim=1)

class SegmentationHead(nn.Sequential):
    """Segmentation head for a segmentation model"""
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity() # can be removed
        activation = nn.Identity() # can be removed
        super().__init__(conv2d, upsampling, activation)

class RuleSegmentationHead(nn.Module):
    """
    Head taking intermediate concept predictions and correction activations as inputs and combining them to obtain 
    a segmentation map
    """
    def __init__(self, in_channels, out_channels, interm_channels, corr_channels, thresholds, rules, act_encoding,
                     kernel_size=3, decision = 'f'):
        """
        Args:
            - in_channels (int): number of input channels, i.e. number of channels in the incoming correction activations
            - out_channels (int): number of output channels
            - interm_channels (int): number of channels in the intermediate concept predictions, i.e. number of intermediate
                concepts
            - corr_channels (int): number of correction channels
            - thresholds (list of list of float/int): list of pre-defined thresholds for each intermediate concept channel
            - rules (1D-Tensor): tensor of integers corresponding to rule categories. If intermediate concepts have
                respectively I and J intervals (separated by thresholds), element (i*J + j) of the tensor, with i in 
                [0, I) and j in [0, J), is the category to assign for intermediate concept values falling in the i-th 
                and the j-th intervals. 
            - act_encoding (2D-Tensor): tensor of hard-coded probability vectors for each unique integer value in 
                'rules' (i.e. each rule category)
            - kernel_size: kernel size and half kernel size for the intermediate concept and correction heads respectively
            - decision: segmentation task configuration. 'f': flat, i.e. one unique task with all classes at the same
                level, or 'h': hierarchical i.e. with several sub-tasks
        """
        super().__init__()

        self.corr_channels = corr_channels
        self.out_channels = out_channels
        # check types and dimensions
        try:
            single_channel = sum(interm_channels) == 1
        except TypeError:
            single_channel = interm_channels == 1
            interm_channels = [interm_channels]
        if single_channel:
            if not isinstance(thresholds, list):
                thresholds = [thresholds]
        else: # check type and length of thresholds
            if not isinstance(thresholds, list):
                raise TypeError('"thresholds" should be a list containing the thresholds for each intermediate variable')
        self.interm_var = len(interm_channels) 
        # register buffers
        self.register_buffer('rules', rules)
        self.register_buffer('act_encoding', act_encoding)
        for i, t in enumerate(thresholds):
            self.register_buffer('thresholds_'+str(i), t)
        n_cat = torch.tensor([len(t) + 1 for t in thresholds])
        self.register_buffer('n_cat', n_cat)

        self.conv_interm = nn.Conv2d(in_channels, sum(interm_channels), kernel_size=kernel_size, 
                            padding=kernel_size // 2)
        
        self.process_interm = self.process_regr_logits 
        self.get_categories = self.regr_to_cat 
        self.conv_corr = nn.Sequential(nn.Conv2d(in_channels+out_channels, in_channels//2, kernel_size=2*kernel_size+1, 
                                                                                                padding=kernel_size),
                                       nn.Conv2d(in_channels//2, corr_channels, kernel_size=2*kernel_size+1, 
                                                                                                padding=kernel_size))
        self.correction_correction = self.apply_additive_correction 

        if decision == 'f':
            self.complete_correction = complete_flat_correction
        else:
            self.complete_correction = complete_hierarchical_correction
            
    def load_buffers(self, buffer_val_list, device):
        self.rules = buffer_val_list[0].to(device)
        self.act_encoding = buffer_val_list[1].to(device) 
        self.n_cat = buffer_val_list[-1].to(device)
        
    
    def regr_to_cat(self, x):
        # the intermediate concept channels are treated separately because they might have a different number of thresholds
        h, w = x.shape[-2:]
        n = x.shape[0]
        cat = torch.empty_like(x, dtype=torch.long)
        for idx, threshold in enumerate(self.thresholds()):
            t = threshold.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, n, h, w)
            comp = x[:, idx] >= t
            cat[:, idx] = torch.sum(comp.long(), axis=0)
        return cat

    def process_regr_logits(self, x):
        return f.relu(x, inplace=False)
    
    def thresholds(self):
        for i in range(self.interm_var):
            yield self.__getattr__('thresholds_'+str(i))

    def intersect(self, cat):
        y = cat[:,-1]
        for i in range(self.interm_var - 2, -1, -1):
            y = y + cat[:, i] * self.n_cat[i+1] 
        y = self.rules[y]
        
        return y

    def apply_additive_correction(self, rule_activations, corr_activations):
        activations = rule_activations + corr_activations 
        return activations

    def forward(self, x):
        x_interm, x_corr = x
        # compute intermediate concept activations
        interm_activations = self.conv_interm(x_interm)
        proc_interm_activations = self.process_interm(interm_activations)
        # apply rules
        rule_cat = self.intersect(self.get_categories(proc_interm_activations))
        # translate categories into hard-coded activations vectors
        rule_activations = self.act_encoding[rule_cat].transpose(2, 3).transpose(1, 2) 
        # compute correction activations
        corr_activations = self.conv_corr(torch.cat((x_corr, rule_activations), dim = 1))
        corr_complete_activations = self.complete_correction(corr_activations)
        # apply correction
        final_activations = self.correction_correction(rule_activations, corr_complete_activations)
        return final_activations, rule_cat, corr_complete_activations, proc_interm_activations


