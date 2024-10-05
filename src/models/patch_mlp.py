import os
import torch
import numpy as np
import json

from torch import nn

from .modules import PatchMLP
from ..dataset.dataloader import *
    
class PatchMLPModel:
    def __init__(self,
            patch_radius: int,
            layers: list[int] = [200]*4,
            ordinal_encoding: bool = False,
            mine_rate_channel: bool = False,
            device: str = 'cpu'):
        self.pad = patch_radius
        self.kernel = 2*patch_radius+1
        self.device = device
        self.layers = layers
        self.ordinal_encoding = ordinal_encoding
        self.mine_rate_channel = mine_rate_channel
        self.transform = GameStateTransform(self.pad, self.ordinal_encoding, self.mine_rate_channel)
        input_mask = torch.ones((self.kernel, self.kernel))
        input_mask[patch_radius, patch_radius] = 0 # mask the value of the cell to predict
        channels = 4 if ordinal_encoding else 12
        channels += 1 if mine_rate_channel else 0
        self.model = PatchMLP(
            in_channels=channels,
            out_channels=1,
            patch_size= self.kernel,
            padding=0,
            layer_units=layers,
            input_mask = input_mask,
            out_activation=nn.Sigmoid()
        )
        self.train_loss_log = []
        self.test_loss_log = []

    def loss(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        squeezed = pred.view(pred.size(0), pred.size(2), pred.size(3))
        return nn.functional.binary_cross_entropy(squeezed, target, weight=weights)
    
    def __call__(self, game_state: np.ndarray, tot_mines: np.ndarray= None) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        x, _, _ = self.transform(game_state, tot_mines)
        x = x.to(self.device)
        return self.model(x).view(game_state.shape).detach().cpu().numpy()
    
    def train(self, dataloader, optimizer):
        self.model.train()
        self.model.to(self.device)
        train_loss = 0
        for batch, (x, y, w) in enumerate(dataloader):
            x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)

            # Compute prediction error
            pred = self.model(x)
            loss = self.loss(pred, y, w)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            
        self.train_loss_log.append(train_loss/len(dataloader))

    def test(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0
        with torch.no_grad():
            for x, y, w in dataloader:
                x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
                pred = self.model(x)
                test_loss += self.loss(pred, y, w).item()
        self.test_loss_log.append(test_loss / len(dataloader))

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
