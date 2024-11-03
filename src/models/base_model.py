import os
import torch
import numpy as np

from torch import nn

class MinesweeperModel:
    def __init__(self, model: nn.Module, device: str):
        self.model = model
        self.device = device

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
        with torch.no_grad():
            for b in range(0, game_state.shape[0], batch_size):
                state = game_state[b:b+batch_size]
                mines_n = tot_mines[b:b+batch_size] if tot_mines is not None else None
                x, _, _ = self.transform(state, mines_n)
                x = x.to(self.device)
                out.append(self.model(x).view(state.shape).detach().cpu().numpy())
        return np.concatenate(out)

    def train(self, dataloader, optimizer) -> float:
        """returns the mean batch loss"""
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

        return train_loss / len(dataloader)

    def test(self, dataloader) -> float:
        """returns the mean batch loss"""
        self.model.eval()
        self.model.to(self.device)
        test_loss = 0
        with torch.no_grad():
            for x, y, w in dataloader:
                x, y, w = x.to(self.device), y.to(self.device), w.to(self.device)
                pred = self.model(x)
                test_loss += self.loss(pred, y, w).item()

        return test_loss / len(dataloader)

    def save(self, path: str):
        """Save model to file"""
        raise NotImplementedError()

    @staticmethod
    def load(path: str, device: str):
        """Load model from file"""
        raise NotImplementedError()