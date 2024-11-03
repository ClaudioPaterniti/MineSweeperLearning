import os
import torch
import json

from torch import nn

from .modules import ConvNet
from ..dataloader.dataloader import *
from .base_model import MinesweeperModel

class ConvModel(MinesweeperModel):
    def __init__(self,
            in_kernel_radius: int,
            layers_channels: list[int],
            use_resblock: bool = True,
            ordinal_encoding: bool = False,
            mine_rate_channel: bool = True,
            device: str = 'cpu'):

        self.in_kernel_radius = in_kernel_radius
        self.ordinal_encoding = ordinal_encoding
        self.mine_rate_channel = mine_rate_channel
        self.layers_channels = layers_channels
        self.use_resblock = use_resblock
        channels = 4 if ordinal_encoding else 12
        if mine_rate_channel:
            channels += 1

        model = ConvNet(
            in_channels= channels,
            in_padding = 0,
            in_kernel=2*in_kernel_radius + 1,
            out_channels=1,
            layers_channels=layers_channels,
            use_resblock=use_resblock,
            out_activation=nn.Sigmoid()
        )
        super().__init__(model, device)

        self.transform = GameStateTransform(
            in_kernel_radius, self.ordinal_encoding, self.mine_rate_channel)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        filename, _ = os.path.splitext(path)
        meta = {
            'inKernelRadius': self.in_kernel_radius,
            'layersChannels': self.layers_channels,
            'useResblock': self.use_resblock,
            'ordinalEncoding': self.ordinal_encoding,
            'mineRateChannel': self.mine_rate_channel,
        }
        with open(filename+'.json', 'w+') as f:
            json.dump(meta, f, indent=4)

    @staticmethod
    def load(path: str, device: str):
        filename, _ = os.path.splitext(path)
        with open(filename+'.json', 'r') as f:
            meta: dict = json.load(f)
        model = ConvModel(
            in_kernel_radius=meta['inKernelRadius'],
            layers_channels=meta['layersChannels'],
            use_resblock=meta['useResblock'],
            ordinal_encoding=meta.get('ordinalEncoding'),
            mine_rate_channel=meta.get('mineRateChannel'),
            device=device)
        model.model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        return model
