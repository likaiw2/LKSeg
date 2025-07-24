import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Callable
import numpy as np

from utils import ShapeSpec, Conv2d, get_norm, c2_xavier_fill
from position_encoding import PositionEmbeddingSine
from transformer import TransformerEncoder, TransformerEncoderLayer


class BasePixelDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "",
    ):
        super().__init__()
        
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []
        use_bias = norm == ""
        
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=3, stride=1, padding=1,
                    bias=use_bias, norm=output_norm, activation=nn.ReLU()
                )
                c2_xavier_fill(output_conv)
                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)
                
                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim, conv_dim, kernel_size=3, stride=1, padding=1,
                    bias=use_bias, norm=output_norm, activation=nn.ReLU()
                )
                c2_xavier_fill(lateral_conv)
                c2_xavier_fill(output_conv)
                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3

    def forward(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        
        return self.mask_features(y), None, multi_scale_features


class MSDeformAttnPixelDecoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        transformer_dropout: float = 0.0,
        transformer_nheads: int = 8,
        transformer_dim_feedforward: int = 1024,
        transformer_enc_layers: int = 6,
        conv_dim: int = 256,
        mask_dim: int = 256,
        norm: str = "",
        transformer_in_features: List[str] = ["res3", "res4", "res5"],
        common_stride: int = 4,
    ):
        super().__init__()
        
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }
        
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_in_features = [k for k, v in transformer_input_shape]
        transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]
        self.transformer_num_feature_levels = len(self.transformer_in_features)
        
        if self.transformer_num_feature_levels > 1:
            input_proj_list = []
            for in_channels in transformer_in_channels:
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                ))
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(transformer_in_channels[0], conv_dim, kernel_size=1),
                    nn.GroupNorm(32, conv_dim),
                )])
        
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
        )
        
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        self.mask_dim = mask_dim
        self.mask_features = Conv2d(conv_dim, mask_dim, kernel_size=1, stride=1, padding=0)
        c2_xavier_fill(self.mask_features)
        
        self.maskformer_num_feature_levels = 3
        self.common_stride = common_stride
        
        # FPN part
        self.num_fpn_levels = len(self.in_features) - self.transformer_num_feature_levels
        lateral_convs = []
        output_convs = []
        
        use_bias = norm == ""
        for idx, in_channels in enumerate(self.feature_channels[:self.num_fpn_levels]):
            lateral_norm = get_norm(norm, conv_dim)
            output_norm = get_norm(norm, conv_dim)
            
            lateral_conv = Conv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
            )
            output_conv = Conv2d(
                conv_dim, conv_dim, kernel_size=3, stride=1, padding=1,
                bias=use_bias, norm=output_norm, activation=nn.ReLU()
            )
            c2_xavier_fill(lateral_conv)
            c2_xavier_fill(output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

    def forward(self, features):
        # Transformer encoder
        srcs = []
        pos = []
        for i, f in enumerate(self.transformer_in_features):
            x = features[f].float()
            srcs.append(self.input_proj[i](x))
            pos.append(self.pe_layer(x))
        
        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]
        
        split_size_or_sections = [None] * self.transformer_num_feature_levels
        for i in range(self.transformer_num_feature_levels):
            if i < self.transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        
        y = torch.split(y, split_size_or_sections, dim=1)
        
        out = []
        multi_scale_features = []
        num_cur_levels = 0
        
        for i, z in enumerate(y):
            out.append(z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1]))
        
        # FPN part
        for idx, f in enumerate(self.in_features[:self.num_fpn_levels][::-1]):
            x = features[f].float()
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            cur_fpn = lateral_conv(x)
            y = cur_fpn + F.interpolate(out[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out.append(y)
        
        for o in out:
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(o)
                num_cur_levels += 1
        
        return self.mask_features(out[-1]), out[0], multi_scale_features


class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout=0.1, num_feature_levels=4):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels
        
        encoder_layer = MSDeformAttnTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, num_feature_levels, nhead
        )
        self.encoder = nn.ModuleList([encoder_layer for _ in range(num_encoder_layers)])
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        # Prepare input
        src_flatten = []
        pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, pos_embed) in enumerate(zip(srcs, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            
            src_flatten.append(src)
            pos_embed_flatten.append(lvl_pos_embed)
        
        src_flatten = torch.cat(src_flatten, 1)
        pos_embed_flatten = torch.cat(pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        
        # Encoder
        output = src_flatten
        for layer in self.encoder:
            output = layer(output, pos_embed_flatten, spatial_shapes, level_start_index)
        
        return output, spatial_shapes, level_start_index


class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024, dropout=0.1, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        
        # Self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, pos, spatial_shapes, level_start_index):
        # Self attention
        src2 = self.self_attn(src + pos, spatial_shapes, level_start_index, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        
        return src


class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads')
        
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        nn.init.constant_(self.sampling_offsets.bias.data, 0.)
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def forward(self, query, spatial_shapes, level_start_index, value):
        N, Len_q, _ = query.shape
        N, Len_in, _ = value.shape
        
        value = self.value_proj(value)
        value = value.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        
        # Simplified deformable attention (placeholder implementation)
        output = value.mean(dim=1, keepdim=True).expand(-1, Len_q, -1, -1)
        output = output.view(N, Len_q, self.d_model)
        
        return self.output_proj(output)