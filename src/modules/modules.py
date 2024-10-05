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
    
class MaskedConv(nn.Conv2d):
    """
    Partial convolution with some input masked in each patch
    """
    def __init__(self, patch_mask: torch.Tensor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            'kernel_input_mask', patch_mask.to(dtype=self.weight.dtype), persistent=False)

    def forward(self, x):
        if self.kernel_input_mask is not None:
            with torch.no_grad():
                self.weight *= self.kernel_input_mask
        return super().forward(x)

class PatchMLP(nn.Module):
    """
    Apply a FCMLP patch by patch. Output channels only depends on the inputs of the patch around them
    """
    def __init__(self,
                 in_channels, out_channels, patch_size: int, padding: int,
                 layer_units: list[int], input_mask: torch.Tensor = None,
                 out_activation: nn.Module = None):
        """:param input_mask: (h,w) binary mask to mask some input in the patch"""
        super().__init__()
        in_units = layer_units[0]
        self.conv_in = MaskedConv(input_mask, in_channels, in_units, patch_size, padding=padding)
        self.norm_in = nn.BatchNorm2d(in_units)
        self.relu = nn.ReLU()
        mid = [nn.Identity()]
        for units in layer_units[1:]:
            mid.append(ConvResBlock(in_units, units, True))
            in_units = units
        self.res_stack = nn.Sequential(*mid)
        self.conv_out = nn.Conv2d(in_units, out_channels, 1)
        self.out_activation = out_activation if out_activation else nn.Identity()

    def forward(self, x):
        h = self.conv_in(x)
        h = self.norm_in(h)
        h = self.relu(h)
        h = self.res_stack(h)
        h = self.conv_out(h)
        return self.out_activation(h)