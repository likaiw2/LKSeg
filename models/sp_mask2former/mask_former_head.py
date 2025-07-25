import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from pixel_decoder import BasePixelDecoder, MSDeformAttnPixelDecoder
from transformer_decoder import StandardTransformerDecoder, MultiScaleMaskedTransformerDecoder
from utils import ShapeSpec


class SPMaskFormerHead(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        num_classes: int,
        pixel_decoder: nn.Module,
        transformer_decoder: nn.Module,
        transformer_in_feature: str = "multi_scale_pixel_decoder",
    ):
        super().__init__()
        
        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_decoder
        self.transformer_in_feature = transformer_in_feature
        self.num_classes = num_classes

    def forward(self, features, sp_input=None, targets=None):
        return self.layers(features, sp_input)

    def layers(self, features, sp_input, mask=None):
        mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder(features)
        
        if self.transformer_in_feature == "multi_scale_pixel_decoder":
            predictions = self.predictor(multi_scale_features, mask_features, mask, sp_input)
        elif self.transformer_in_feature == "transformer_encoder":
            assert (
                transformer_encoder_features is not None
            ), "Please use the TransformerEncoderPixelDecoder."
            predictions = self.predictor(transformer_encoder_features, mask_features, mask)
        elif self.transformer_in_feature == "pixel_embedding":
            predictions = self.predictor(mask_features, mask_features, mask)
        else:
            predictions = self.predictor(features[self.transformer_in_feature], mask_features, mask)
        
        return predictions