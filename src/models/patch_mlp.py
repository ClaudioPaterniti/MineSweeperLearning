import os
import torch
import json

from torch import nn

from .modules import PatchMLP
from ..dataloader.dataloader import *
from .base_model import MinesweeperModel
    
class PatchMLPModel(MinesweeperModel):
    def __init__(self,
            patch_radius: int,
            layers: list[int] = [200]*4,
            ordinal_encoding: bool = False,
            mine_rate_channel: bool = True,
            device: str = 'cpu'):
        self.pad = patch_radius
        self.kernel = 2*patch_radius+1
        self.layers = layers
        self.ordinal_encoding = ordinal_encoding
        self.mine_rate_channel = mine_rate_channel
        input_mask = torch.ones((self.kernel, self.kernel))
        input_mask[patch_radius, patch_radius] = 0 # mask the value of the cell to predict
        channels = 4 if ordinal_encoding else 12
        if mine_rate_channel:
            channels += 1

        model = PatchMLP(
            in_channels=channels,
            out_channels=1,
            patch_size= self.kernel,
            padding=0, # input already padded by transform
            layer_units=layers,
            input_mask = input_mask,
            out_activation=nn.Sigmoid()
        )
        super().__init__(model, device)
        
        self.transform = GameStateTransform(self.pad, self.ordinal_encoding, self.mine_rate_channel)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        filename, _ = os.path.splitext(path)
        meta = {
            'patchRadius': self.pad,
            'layers': self.layers,
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
        model = PatchMLPModel(
            patch_radius=meta['patchRadius'],
            layers=meta['layers'],
            ordinal_encoding=meta.get('ordinalEncoding'),
            mine_rate_channel=meta.get('mineRateChannel'),
            device=device)
        model.model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        return model
