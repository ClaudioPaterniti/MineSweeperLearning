import torch
import numpy as np

from typing import Union
from torch import nn
from torch.utils.data import Dataset

from ..game import Game

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
        """Prepare the game state for the model input

        :param padding: the padding to use for cells outside the game
        :param ordinal_encoding: if True numbers are encoded from 0 to 9, otherwise one hot encoding
        :param mines_rate_channel: if True the model requires the tot_mine input
        and a constant channel (closed-flags)/tot_mines is added.
        """
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
        """11 channels: numbers 0 to 8, 9 for closed cells, 10 for flags"""
        x = torch.from_numpy(state).long() # long required by one_hot function
        permute = (0, 3, 1, 2) if len(state.shape) == 3 else (2, 0, 1)
        x = nn.functional.one_hot(x, 11).permute(*permute) # move the encoding from last to channel dimension
        return nn.functional.pad(x, [self.padding]*4)
    
    def _ordinal_encoding(self, state: np.ndarray) -> torch.Tensor:
        """3 channels: 0-9 numbers shifted of 1 with 0 for non-numbers, binary closed cells and flags"""
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