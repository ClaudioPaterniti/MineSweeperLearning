import os
import torch
import numpy as np
import json

from torch import nn

from .modules import PatchMLP
from ..dataloader.dataloader import *

class MinesweeperModel:
    def __init__(self, model: nn.Module):
        self.model = model
        self.train_loss_log = []
        self.test_loss_log = []

    def loss(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        squeezed = pred.view(pred.size(0), pred.size(2), pred.size(3))
        return nn.functional.binary_cross_entropy(squeezed, target, weight=weights)
    
    def transform(self,
            state: np.ndarray,
            tot_mines: np.ndarray = None,
            mines: np.ndarray = None,
            weights: np.ndarray = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare input for the model
        
        :param state: (n,w,h) with games state
        :param tot_mines: (n) with tot number of mines in the game
        :param mines: binary (n,w,h) with the mines positions
        :param weights: (n,w,h) with the weights for the cell loss
        
        :returns: (x, y, w) = model input, target and loss weights"""
        raise NotImplementedError()
    
    def __call__(self,
            game_state: np.ndarray, tot_mines: np.ndarray= None,
            batch_size: int = 1000, **kwargs) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        out = []
        for b in range(0, game_state.shape[0], batch_size):
            state = game_state[b:b+batch_size]
            mines_n = tot_mines[b:b+batch_size] if tot_mines is not None else None
            x, _, _ = self.transform(state, mines_n)
            x = x.to(self.device)
            out.append(self.model(x).view(state.shape).detach().cpu().numpy())
        return np.concatenate(out)
    
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
    
class PatchMLPModel(MinesweeperModel):
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
        super().__init__(model)
        
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
