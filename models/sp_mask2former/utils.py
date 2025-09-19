import torch
import torch.nn as nn
from typing import Optional, Union, Callable


class ShapeSpec:
    """Simple ShapeSpec replacement"""
    def __init__(self, channels: int, stride: int):
        self.channels = channels
        self.stride = stride


def get_norm(norm_type: str, channels: int) -> Optional[nn.Module]:
    """Get normalization layer"""
    if norm_type == "BN":
        return nn.BatchNorm2d(channels)
    elif norm_type == "GN":
        return nn.GroupNorm(32, channels)
    elif norm_type == "":
        return None
    return None


def Conv2d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, 
           padding: int = 0, bias: bool = True, norm: Optional[nn.Module] = None, 
           activation: Optional[nn.Module] = None) -> nn.Module:
    """Conv2d with optional normalization and activation"""
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
    
    layers = [conv]
    if norm is not None:
        layers.append(norm)
    if activation is not None:
        layers.append(activation)
    
    if len(layers) == 1:
        return conv
    return nn.Sequential(*layers)


def c2_xavier_fill(module: nn.Module):
    """
    Initialize `module.weight` using the "XavierFill" implemented in Caffe2.
    Also initializes `module.bias` to 0.
    """
    if hasattr(module, 'weight') and module.weight is not None:
        if module.weight.dim() >= 2:
            nn.init.xavier_uniform_(module.weight)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, 0)
