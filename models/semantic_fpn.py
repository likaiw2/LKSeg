import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from segmentation_models_pytorch.encoders import get_encoder

# =====================================================
# ↓↓↓ 下面是原 seg_base.py 中的核心组件代码 ↓↓↓
# =====================================================

class ConvBlock(nn.Sequential):
    """带有可选 BN 和 ReLU 的标准卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, bn=True, relu=True, init_fn=None):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.ReLU(True) if relu else nn.Identity()
        )
        if init_fn:
            self.apply(init_fn)

    @staticmethod
    def same_padding(kernel_size, dilation):
        # 保证 'same' 卷积
        return dilation * (kernel_size - 1) // 2


def init_conv(m):
    """Kaiming 初始化"""
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, a=1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


def conv_with_kaiming_uniform(use_bn=False, use_relu=False):
    """返回一个构造 ConvBlock 的函数，支持自定义 BN/ReLU"""
    def make_conv(in_channels, out_channels, kernel_size, stride=1, dilation=1):
        return ConvBlock(
            in_channels, out_channels, kernel_size, stride,
            padding=ConvBlock.same_padding(kernel_size, dilation),
            dilation=dilation, bias=True,
            bn=use_bn, relu=use_relu,
            init_fn=init_conv
        )
    return make_conv

# 默认的 conv_block：带 BN、不带 ReLU
default_conv_block = conv_with_kaiming_uniform(use_bn=True, use_relu=False)


class LastLevelMaxPool(nn.Module):
    """FPN 额外的 MaxPool 生成顶层特征"""
    def forward(self, x):
        return [F.max_pool2d(x, 1, 2, 0)]


class LastLevelP6P7(nn.Module):
    """RetinaNet 用于生成 P6、P7"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.p6 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = nn.Conv2d(out_channels, out_channels, 3, 2, 1)
        for m in (self.p6, self.p7):
            nn.init.kaiming_uniform_(m.weight, a=1)
            nn.init.constant_(m.bias, 0)
        # 如果输入特征跟输出同通道数，则 P6 输入为 P5
        self.use_P5 = (in_channels == out_channels)

    def forward(self, c5, p5):
        x = p5 if self.use_P5 else c5
        p6 = self.p6(x)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


class FPN(nn.Module):
    """
    Feature Pyramid Network：融合多尺度骨干特征
    """
    def __init__(self,
                 in_channels_list,
                 out_channels,
                 conv_block=default_conv_block,
                 top_blocks=None):
        super().__init__()
        self.inner_blocks = []
        self.layer_blocks = []

        # 为每个输入尺度创建 lateral+output conv
        for idx, in_ch in enumerate(in_channels_list, start=1):
            if in_ch == 0:
                continue
            inner_name = f'fpn_inner{idx}'
            layer_name = f'fpn_layer{idx}'
            inner = conv_block(in_ch, out_channels, 1)
            layer = conv_block(out_channels, out_channels, 3, 1)
            self.add_module(inner_name, inner)
            self.add_module(layer_name, layer)
            self.inner_blocks.append(inner_name)
            self.layer_blocks.append(layer_name)

        self.top_blocks = top_blocks

    def forward(self, inputs):
        # 最顶层
        last_inner = getattr(self, self.inner_blocks[-1])(inputs[-1])
        results = [getattr(self, self.layer_blocks[-1])(last_inner)]

        # 自顶向下融合
        for feat, inner_name, layer_name in zip(
                inputs[-2::-1], self.inner_blocks[-2::-1], self.layer_blocks[-2::-1]):
            lateral = getattr(self, inner_name)(feat)
            # 上采样上一层
            if last_inner.shape[-1] != lateral.shape[-1]:
                last_inner = F.interpolate(last_inner, scale_factor=2,
                                           mode='bilinear', align_corners=False)
            last_inner = lateral + last_inner
            results.insert(0, getattr(self, layer_name)(last_inner))

        # 可选的额外层（P6/P7 或 MaxPool）
        if isinstance(self.top_blocks, LastLevelP6P7):
            extra = self.top_blocks(inputs[-1], results[-1])
            results.extend(extra)
        elif isinstance(self.top_blocks, LastLevelMaxPool):
            extra = self.top_blocks(results[-1])
            results.extend(extra)

        return tuple(results)


class AssymetricDecoder(nn.Module):
    """
    非对称解码器：将多尺度特征上采样并累加
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super().__init__()
        # 配置 Norm 参数
        if norm_fn is nn.BatchNorm2d:
            norm_args = dict(num_features=out_channels)
        elif norm_fn is nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('使用 GroupNorm 时必须指定 num_groups_gn')
            norm_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError(f'不支持的 norm 类型 {norm_fn}')

        # 为每个输入尺度构建一系列 Conv→BN→ReLU→Upsample
        self.blocks = nn.ModuleList()
        for os in in_feat_output_strides:
            num_up = int(math.log2(os)) - int(math.log2(out_feat_output_stride))
            layers = []
            for i in range(max(num_up, 1)):
                in_ch = in_channels if i == 0 else out_channels
                layers.append(nn.Conv2d(in_ch, out_channels, 3, 1, 1, bias=False))
                layers.append(norm_fn(**norm_args))
                layers.append(nn.ReLU(inplace=True))
                if num_up > 0:
                    layers.append(nn.UpsamplingBilinear2d(scale_factor=2))
            self.blocks.append(nn.Sequential(*layers))

    def forward(self, feats: list):
        outs = []
        for feat, block in zip(feats, self.blocks):
            outs.append(block(feat))
        # 对所有尺度的特征求平均
        out = sum(outs) / len(outs)
        return out

# =====================================================
# ↑↑↑ seg_base.py 核心组件结束 ↑↑↑
# =====================================================


class SemanticFPN(nn.Module):
    """
    将 Encoder → FPN → Decoder → SegHead 串联起来的完整语义分割模型
    """
    def __init__(
        self,
        # Backbone 配置
        encoder_name: str = 'resnet50',
        encoder_weights: str = 'imagenet',
        in_channels: int = 3,
        # FPN 配置
        fpn_in_channels_list: tuple = (256, 512, 1024, 2048),
        fpn_out_channels: int = 256,
        # Decoder 配置
        decoder_in_channels: int = 256,
        decoder_out_channels: int = 128,
        decoder_in_strides: tuple = (4, 8, 16, 32),
        decoder_out_stride: int = 4,
        decoder_norm_fn=nn.BatchNorm2d,
        decoder_num_groups: int = None,
        # SegHead & Loss 配置
        num_classes: int = 7,
        ignore_index: int = 255
    ):
        super().__init__()

        # 1. Backbone：提取特征
        self.encoder = get_encoder(
            name='resnet50',
            weights='imagenet',
            in_channels=3,
            output_stride=32,
        )

        # 2. FPN：融合多尺度特征
        self.fpn = FPN(
            in_channels_list=fpn_in_channels_list,
            out_channels=fpn_out_channels,
            conv_block=default_conv_block,
            top_blocks=None
        )

        # 3. Decoder：上采样并融合
        self.decoder = AssymetricDecoder(
            in_channels=decoder_in_channels,
            out_channels=decoder_out_channels,
            in_feat_output_strides=decoder_in_strides,
            out_feat_output_stride=decoder_out_stride,
            norm_fn=decoder_norm_fn,
            num_groups_gn=decoder_num_groups
        )

        # 4. Segmentation Head：1×1 + 上采样
        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_out_channels, num_classes, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        # 5. 损失函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, x):
        # Backbone
        feats = self.encoder(x)
        if len(feats) == 5:
            feats = feats[1:]  # 丢弃第一个特征层

        # FPN
        pyramid_feats = self.fpn(feats)

        # Decoder
        decoded = self.decoder(pyramid_feats)

        # SegHead
        logits = self.seg_head(decoded)
        
        return logits