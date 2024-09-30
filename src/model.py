import torch
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader

from .game import Game
from .modules.modules import PatchMLP


class MineSweeperDataset(Dataset):

    def __init__(self, games: Game, transform=None):
        self.games = games
        self.transform = transform

    def __len__(self):
        return self.games.n

    def __getitem__(self, idx):
        numbers, closed, flags = self.games.visible_numbers()[idx], (1-self.games.open_cells[idx]), self.games.flags[idx]
        # closed cells start at -1, +10*closed set them to 9, all flag are closed and are thus set to 10
        sample = numbers + 10*closed + flags
        mines_n = self.games.mines_n
        target = self.games.mines[idx]

        if self.transform:
            sample, target = self.transform(sample, mines_n, target)

        return sample, target
    

class OnHotEncodingTransform:
    def __init__(self, padding: int):
        self.padding = padding

    def __call__(self, numbers: np.ndarray, mines_n: int, mines: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(numbers).long() # long required by one_hot function
        y = torch.from_numpy(mines)
        # 11 classes: numbers 0 to 8, 9 for closed cells, 10 for flags
        x = nn.functional.one_hot(x, 11).permute(2, 0, 1) # move the encoding from last to channel dimension
        x = nn.functional.pad(x, [self.padding]*4)
        # padding mask is the 12th channel, indicating cells outside the grid
        padding_mask = 1-nn.functional.pad(torch.ones(numbers.shape), [self.padding]*4)
        x = torch.concat((x, torch.unsqueeze(padding_mask, 0)))
        return x, y
    
class PatchMLPModel:
    def __init__(self, patch_radius: int):
        self.pad = patch_radius
        self.model = PatchMLP(
            in_channels=12,
            out_channels=1,
            patch_size=2*patch_radius+1,
            padding=0,
            layer_units=[250]*6,
            out_activation=nn.Sigmoid()
        )
        self.bce = nn.BCELoss()
        self.train_loss_log = []
        self.test_loss_log = []

    def loss(self, x: torch.Tensor, pred: torch.Tensor, target: torch.Tensor):
        closed = x[:, 9, self.pad:-self.pad, self.pad:-self.pad] # closed cells
        flags = x[:, 10, self.pad:-self.pad, self.pad:-self.pad] # flags
        squeezed = pred.view(pred.size(0), pred.size(2), pred.size(3))
        return self.bce(squeezed*closed*(1-flags), target*(1-flags)) # we ignore the loss on open cells and flags
    
    def train(self, dataloader, optimizer, device):
        self.model.train()
        train_loss = 0
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Compute prediction error
            pred = self.model(x)
            loss = self.loss(x, pred, y)

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            
        self.train_loss_log.append(train_loss/len(dataloader))

    def test(self, dataloader, device):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.model(X)
                test_loss += self.loss(X, pred, y).item()
        self.test_loss_log.append(test_loss / len(dataloader))
