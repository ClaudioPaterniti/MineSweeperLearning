import os
import torch
import json

from torch import nn

from .modules import Unet
from ..dataloader.dataloader import *
from .base_model import MinesweeperModel
    
class UnetModel(MinesweeperModel):
    def __init__(self,
            map_size: tuple[int, int],
            decoder_shapes: list[tuple[int, int, int]],
            use_conv: bool = False,            
            use_resblock: bool = False,
            ordinal_encoding: bool = False,
            mine_rate_channel: bool = True,
            device: str = 'cpu'):
       
        self.device = device
        self.ordinal_encoding = ordinal_encoding
        self.mine_rate_channel = mine_rate_channel
        self.map_size = tuple(map_size)
        self.decoder_shapes = decoder_shapes
        self.use_conv = use_conv
        self.use_resblock = use_resblock
        channels = 4 if ordinal_encoding else 12
        if mine_rate_channel:
            channels += 1

        model = Unet(
            input_shape=(channels, map_size[0] + 1, map_size[1] + 1),
            decoder_shapes=decoder_shapes,
            in_padding=0,
            out_channels=1,
            out_activation=nn.Sigmoid(),
            use_conv=use_conv,
            use_resblock=use_resblock
        )
        super().__init__(model)
        
        self.transform = GameStateTransform(1, self.ordinal_encoding, self.mine_rate_channel)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        filename, _ = os.path.splitext(path)
        meta = {
            'mapSize': self.map_size,
            'decoderShapes': self.decoder_shapes,
            'useConv': self.use_conv,
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
        model = UnetModel(
            map_size=meta['mapSize'],
            decoder_shapes=meta['decoderShapes'],
            use_conv=meta['useConv'],
            use_resblock=meta['useResblock'],
            ordinal_encoding=meta.get('ordinalEncoding'),
            mine_rate_channel=meta.get('mineRateChannel'),
            device=device)
        model.model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
        return model
