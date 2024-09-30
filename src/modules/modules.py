import torch
from torch import nn

class ConvResBlock(nn.Module):
    """
    Residual block with convolution, maintains the input img shape (H,W)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel1x1: bool = False):
        super().__init__()
        kernel_size, padding = (1, 0) if kernel1x1 else (3, 1)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_skip = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding) if in_channels != out_channels else nn.Identity()
        self.norm_out =  nn.BatchNorm2d(out_channels)
        self.norm_in = nn.BatchNorm2d(in_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.norm_in(x)
        h = self.conv_in(x)
        h = self.activation(h)
        h = self.norm_out(h)
        h = self.conv_out(h)
        skip = self.conv_skip(x)
        return self.activation(h + skip)
    

class PatchMLP(nn.Module):
    """
    Apply a FCMLP patch by patch. Output channels only depends on the inputs of the patch around them
    """
    def __init__(self,
                 in_channels, out_channels, patch_size: int, padding: int,
                 layer_units: list[int], out_activation: nn.Module = None):
        super().__init__()
        in_units = layer_units[0]
        self.conv_in = nn.Conv2d(in_channels, in_units, patch_size, padding=padding)
        mid = [nn.Identity()]
        for units in layer_units[1:]:
            mid.append(ConvResBlock(in_units, units, True))
            in_units = units
        self.res_stack = nn.Sequential(*mid)
        self.conv_out = nn.Conv2d(in_units, out_channels, 1)
        self.out_activation = out_activation if out_activation else nn.Identity()

    def forward(self, x):
        h = self.conv_in(x)
        h = nn.ReLU()(h)
        h = self.res_stack(h)
        h = self.conv_out(h)
        return self.out_activation(h)