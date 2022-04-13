import torch
from . import initialization as init

class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)

    def forward(self, *x):
        """
        Pass x through the model's encoder, decoder and heads
        """
        features = self.encoder(*x)
        decoder_output = self.decoder(*features)
        output = self.segmentation_head(decoder_output)

        return output

