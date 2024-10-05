import os
import torch
import numpy as np
import json

from typing import Union
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .game import Game
from .modules.modules import PatchMLP


class MineSweeperDataset(Dataset):

    def __init__(self,
                 games: Game, transform=None,
                 mines_weight: int = 1,
                 losing_moves_weight: int = 1,):
        self.transform = transform
        self.target = games.mines
        self.states = games.game_state()
        self.tot_mines = games.mines.sum(axis=(-1,-2))
        self.losing_moves_weight = losing_moves_weight
        self.mines_weight = mines_weight
        self.weights = self._compute_weights(games)
        self.rng = np.random.default_rng()

    # weights should be recomputed after training on them
    def mix(self, games: Game):
        shuffle = self.rng.permutation(self.target.shape[0])
        self.target = np.concatenate((self.target[shuffle[:-games.n]], games.mines))
        self.states = np.concatenate((self.states[shuffle[:-games.n]], games.game_state()))
        weights = self._compute_weights(games)
        self.weights = np.concatenate((self.weights[shuffle[:-games.n]], weights))

    def _compute_weights(self, games: Game) -> np.ndarray:
        return (1
                + games.mines*(self.mines_weight-1)
                + games.losing_moves()*(self.losing_moves_weight-1))

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        sample = self.states[idx]
        target = self.target[idx]
        weights = self.weights[idx]
        tot_mines = self.tot_mines[idx]

        if self.transform:
            sample, target, weights = self.transform(sample, tot_mines, target, weights)

        return sample, target, weights    

class GameStateTransform:
    def __init__(self,
            padding: int, ordinal_encoding: bool = False, mines_rate_channel: bool= True):
        self.padding = padding
        self.ordinal_encoding = ordinal_encoding
        self.mines_rate_channel = mines_rate_channel

    def __call__(self,
            state: np.ndarray,
            tot_mines: np.ndarray = None,
            mines: np.ndarray = None,
            weights: np.ndarray = None) -> torch.Tensor:
        # numbers can be (h,w) or (n,h,w), tot_mines scalar or (n)
        if self.mines_rate_channel and tot_mines is None:
            raise Exception(f'This model requires the total number of mines as input')
        channels = [] # list of channel to concat
        if self.ordinal_encoding:
            channels.append(self._ordinal_encoding(state))
        else:
            channels.append(self._one_hot_encoding(state))
        y = torch.from_numpy(mines).float() if mines is not None else None
        w = torch.from_numpy(weights).float() if weights is not None else None
        concat_dim = 1 if len(state.shape) == 3 else 0
        # padding mask channel, indicating cells outside the grid
        padding_mask = 1-nn.functional.pad(torch.ones(state.shape), [self.padding]*4)
        channels.append(padding_mask.unsqueeze(concat_dim))
        if self.mines_rate_channel: # add constant channels with the rate mines/closed
            channels.append(
                self._mine_rate_channel(state, tot_mines, padding_mask.shape).unsqueeze(concat_dim))
        x = torch.concat(channels, dim=concat_dim)
        return x, y, w
    
    def _one_hot_encoding(self, state: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(state).long() # long required by one_hot function
        permute = (0, 3, 1, 2) if len(state.shape) == 3 else (2, 0, 1)
        # 11 classes: numbers 0 to 8, 9 for closed cells, 10 for flags
        x = nn.functional.one_hot(x, 11).permute(*permute) # move the encoding from last to channel dimension
        return nn.functional.pad(x, [self.padding]*4)
    
    def _ordinal_encoding(self, state: np.ndarray) -> torch.Tensor:
        numbers = (state + 1)*(state < 9) # shift numbers of 1 to place 0 on non-number squares
        flags = state == 10
        closed = state == 9
        channels = np.stack((numbers, flags, closed), axis = 1 if len(state.shape) == 3 else 0)
        return nn.functional.pad(torch.from_numpy(channels).float(), [self.padding]*4)

    def _mine_rate_channel(self,
            state: np.ndarray, tot_mines: Union[int,np.ndarray], shape: tuple[int]) -> torch.Tensor:
        reshape = (-1,1,1) if len(state.shape) == 3 else (-1,1,)
        rates = (state == 9).sum(axis=(-2,-1))/tot_mines
        rate_channel = np.broadcast_to(rates.reshape(reshape), shape)
        return torch.from_numpy(rate_channel.copy()).float()
    
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
