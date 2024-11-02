import numpy as np
import torch
from torch import nn

class Conv3x3Block(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.norm(x)
        h = self.conv(h)
        return self.relu(h)

class ConvResBlock(nn.Module):
    """
    Residual block with convolution, maintains the input img shape (H,W)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel1x1: bool = False):
        super().__init__()
        kernel_size, padding = (1, 0) if kernel1x1 else (3, 1)
        self.conv_in = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.conv_out = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.conv_skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
            if in_channels != out_channels else nn.Identity())
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
        """:param patch_mask: (h,w) or (c,h,w) binary tensor"""
        super().__init__(*args, **kwargs)
        self.register_buffer(
            'kernel_input_mask', patch_mask.to(dtype=self.weight.dtype), persistent=False)

    def forward(self, x):
        if self.kernel_input_mask is not None:
            with torch.no_grad():
                self.weight *= self.kernel_input_mask
        return super().forward(x)

class DownSample(nn.Module):

    def __init__(self, channels: int, in_shape: tuple[int, int], out_shape: tuple[int, int],
            use_conv: bool = False, out_channels: int = None):
        super().__init__()
        if not out_channels: out_channels = channels
        in_shape, out_shape = np.array(in_shape), np.array(out_shape)
        r = in_shape%out_shape
        self.padding = (r > 1)*(out_shape - r + 1)//2
        pad_shape = in_shape + 2*self.padding
        self.stride = pad_shape//out_shape
        self.kernel = self.stride + pad_shape%out_shape
        if use_conv: self.module = nn.Conv2d(
                channels, out_channels, tuple(self.kernel), tuple(self.stride), tuple(self.padding))
        else:
            self.module = nn.MaxPool2d(tuple(self.kernel), tuple(self.stride), tuple(self.padding))

    def forward(self, x):
        return self.module(x)

class UpSample(nn.Module):

    def __init__(self,
            in_channels: int, out_channels: int, in_shape: tuple[int, int], out_shape: tuple[int, int]):
        super().__init__()
        out_shape, in_shape = np.array(out_shape), np.array(in_shape)
        r = out_shape%in_shape
        self.padding = (r > 1)*(in_shape - r + 1)//2
        pad_shape = out_shape + 2*self.padding
        self.stride = pad_shape//in_shape
        self.kernel = self.stride + pad_shape%in_shape
        self.module = nn.ConvTranspose2d(
                in_channels, out_channels, tuple(self.kernel), tuple(self.stride),
                output_padding=tuple(self.padding))

    def forward(self, x):
        return self.module(x)

class PatchMLP(nn.Module):
    """
    Apply a FCMLP patch by patch. Output channels only depends on the inputs of the patch around them
    """
    def __init__(self,
                 in_channels: int, out_channels: int, patch_size: int, padding: int,
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


class ConvNet(nn.Module):
    def __init__(self,
                 in_channels: int, in_padding: int, in_kernel: int, out_channels: int,
                 layers_channels: list[int], use_resblock: bool = True, out_activation: nn.Module = None):
        """:param input_mask: (h,w) binary mask to mask some input in the patch"""
        super().__init__()
        current_channels = layers_channels[0]
        self.conv_in = nn.Conv2d(in_channels, current_channels, in_kernel, 1, in_padding)
        self.relu = nn.ReLU()
        layers = []
        for channels in layers_channels[1:]:
            if use_resblock:
                layers.append(ConvResBlock(current_channels, channels))
            else:
                layers.extend([
                    Conv3x3Block(current_channels, channels),
                    Conv3x3Block(channels, channels)
                ])
            current_channels = channels
        self.layers = nn.Sequential(*layers)
        self.conv_out = nn.Conv2d(current_channels, out_channels, 1)
        self.out_activation = out_activation if out_activation else nn.Identity()

    def forward(self, x):
        h = self.conv_in(x)
        h = self.relu(h)
        h = self.layers(h)
        h = self.conv_out(h)
        return self.out_activation(h)


class Unet(nn.Module):
    def __init__(self,
            input_shape: tuple[int, int, int], decoder_shapes: list[tuple[int, int, int]],
            in_padding: int = 1, out_channels: int = 1, out_activation: nn.Module = None,
            conv_downsample: bool = False, use_resblock: bool = False):
        super().__init__()
        current_shape = decoder_shapes[0]
        self.in_block = nn.Sequential(
            nn.Conv2d(input_shape[0], current_shape[0], 3, padding=in_padding),
            nn.ReLU(),
            Conv3x3Block(current_shape[0], current_shape[0])
        )
        self.conv_out = nn.Conv2d(current_shape[0], out_channels, 1)
        self.out_activation = out_activation if out_activation else nn.Identity()

        self.encoder = nn.ModuleList()
        for shape in decoder_shapes[1:]:
            size = shape[1:]
            channels = shape[0]
            block = [DownSample(current_shape[0], current_shape[1:], size, conv_downsample)]
            if use_resblock:
                block.append(ConvResBlock(current_shape[0], channels))
            else:
                block.extend([
                    Conv3x3Block(current_shape[0], channels),
                    Conv3x3Block(channels, channels)
                ])
            self.encoder.append(nn.Sequential(*block))
            current_shape = shape

        self.decoder = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for shape in reversed(decoder_shapes[:-1]):
            size = shape[1:]
            channels = shape[0]
            block = []
            self.upsamplers.append(
                UpSample(current_shape[0], channels, current_shape[1:], size))
            if use_resblock:
                block.append(ConvResBlock(channels*2, channels))
            else:
                block.extend([
                    Conv3x3Block(channels*2, channels),
                    Conv3x3Block(channels, channels)
                ])
                self.decoder.append(nn.Sequential(*block))
                current_shape = shape

    def forward(self, x):
        h = self.in_block(x)
        encodings = [h]
        for block in self.encoder:
            h = block(h)
            encodings.append(h)
        encodings.pop()
        for upsampler, block in zip(self.upsamplers, self.decoder):
            h = upsampler(h)
            h = torch.cat((encodings.pop(), h), dim=1)
            h = block(h)
        h = self.conv_out(h)
        return self.out_activation(h)