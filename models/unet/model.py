from typing import List
from .decoder import UnetDecoder, RuleUnetDecoder
from ..encoders import ResNetEncoder
from ..base import SegmentationModel
from ..base import SegmentationHead, RuleSegmentationHead
from torch import Tensor
import torch

# constants for default arguments
eps, C = 1e-3, 3
PROB_ENCODING = torch.tensor([  [1.0 - 3*eps,   eps,            eps,            eps         ], 
                                [eps,           1.0 - 3*eps,    eps,            eps         ], 
                                [eps,           eps,            1.0 - 3*eps,    eps         ],
                                [eps,           eps,            eps,            1.0 - 3*eps ]])
ACT_ENCODING = torch.log(PROB_ENCODING) + C


class Unet(SegmentationModel):
    """
    Unet model with a ResNet-18-like encoder, and possibly an auxiliary input source
    Adapted from https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/decoders/unet/model.py 
    """

    def __init__(
        self,
        encoder_depth: int = 4,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 1,
        upsample: bool = False,
        aux_in_channels: int = 0,
        aux_in_position: int = 1,
        **kwargs
    ):
        """
        Args:
            - encoder_depth: number of blocks in the ResNet encoder, each block itself 
                containing 2 blocks (Resnet-18). encoder_depth does not include
                    the initial conv and maxpool layers
            - decoder_channels: number of output channels of each decoder layer
            - in_channels: number of channels of the main input
            - classes: number of classes (i.e. number of output channels)
            - upsample: whether to upsample or not the activations at the end of each 
                decoder block. The upsampling is done via a transposed convolution with 
                upsampling factor 2. If a single value is specified the same value will
                be used for all decoder blocks.
            - aux_in_channels: number of channels of the auxiliary input
            - aux_in_position: position of the auxiliary input in the model:
                0: concatenated to the main input before entering the model
                1: before the 1st block of the encoder
                2: before the 2nd block of the encoder
                3: etc.
        """
        
        super().__init__()
        layers, out_channels = self.set_channels(aux_in_channels, aux_in_position, encoder_depth)
        encoder, decoder, segmentation_head = self._get_model_blocks()
        self.encoder = encoder(in_channels = in_channels,
                        aux_in_channels = aux_in_channels,
                        out_channels = out_channels,
                        layers = layers,
                        aux_in_position = aux_in_position)

        self.decoder = decoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            upsample = upsample,
            use_batchnorm=True
        )
        self.segmentation_head = segmentation_head(
            in_channels=decoder_channels[-1], 
            out_channels=classes,
            kernel_size=3,
            **kwargs
        )

        self.initialize()

    def _get_model_blocks(self):
        return ResNetEncoder, UnetDecoder, SegmentationHead

    def set_channels(self, aux_in_channels, aux_in_position, encoder_depth):
        if (aux_in_channels is None) != (aux_in_position is None):
            raise ValueError('aux_in_channels and aux_in_position should be both specified')
        # architecture based on Resnet-18
        out_channels = [64, 64, 128, 256, 512]
        out_channels = out_channels[:encoder_depth+1]
        if aux_in_position is not None:
            out_channels[aux_in_position] += aux_in_channels
        layers = [2] * (len(out_channels)-1)
        return layers, out_channels

class RuleUnet(Unet): 
    """
        Unet model with a ResNet-18-like encoder. Has a main output corresponding to a segmentation task and intermediate 
        outputs corresponding to the regression of intermediate variables. Thresholds are applied to these variables to 
        obtain class probabilities based on rules defined by the thresholds. A correction feature map is used to correct
        these predictions and obtain the main output.
    """
    def __init__(
        self,
        encoder_depth: int = 4,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        in_channels: int = 3,
        classes: int = 4,
        upsample: bool = False,
        aux_in_channels: int = 0,
        aux_in_position: int = 1,
        interm_channels: int = 2, 
        corr_channels: int = 4, 
        thresholds: List[float] = [[0.5], [0.5]], 
        rules: Tensor = torch.tensor([0, 1, 2, 3]),
        act_encoding : Tensor = ACT_ENCODING,
        decision = 'f'
        ):
        """
        Args:
            - interm_channels: number of intermediate variables to predict
            - corr_channels: number of correction channels (depends on the correction strategy, i.e. the desing of the
                rule module). Should be n_classes for element-wise addition module.
            - thresholds (list of list of float/int): list of pre-defined thresholds for each intermediate concept channel
            - rules (1D-Tensor): tensor of integers corresponding to rule categories. If intermediate concepts have
                respectively I and J intervals (separated by thresholds), element (i*J + j) of the tensor, with i in 
                [0, I) and j in [0, J), is the category to assign for intermediate concept values falling in the i-th 
                and the j-th intervals.
            - act_encoding (2D-Tensor): tensor of hard-coded probability vectors for each unique integer value in 
                'rules' (i.e. each rule category)
            - decision: segmentation task configuration. 'f': flat, i.e. one unique task with all classes at the same
                level, or 'h': hierarchical i.e. with several sub-tasks
            - other args: see mother class
        """
        super().__init__(encoder_depth = encoder_depth,
                        decoder_channels = decoder_channels,
                        in_channels = in_channels,
                        classes = classes,
                        upsample = upsample,
                        aux_in_channels = aux_in_channels,
                        aux_in_position = aux_in_position,
                        interm_channels = interm_channels,
                        corr_channels = corr_channels,
                        thresholds = thresholds,
                        rules = rules,
                        act_encoding = act_encoding,
                        decision = decision
                        )

    def _get_model_blocks(self):
        return ResNetEncoder, RuleUnetDecoder, RuleSegmentationHead
    
