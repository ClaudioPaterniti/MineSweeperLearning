import torch
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader

from .game import Game
from .modules.modules import PatchMLP


class MineSweeperDataset(Dataset):

    def __init__(self, games: Game, transform=None, losing_moves_weight: int = 3):
        self.transform = transform
        self.target = games.mines
        self.states = games.game_state()
        self.losing_moves_weight = losing_moves_weight
        # weight ignore the loss on open cells and flags, and multiply loss on last losing moves
        self.weights = self._compute_weights(games)
        self.rng = np.random.default_rng()

    # weights should be recomputed after training on them
    def mix(self, games: Game):
        shuffle = self.rng.permutation(self.target.shape[0])
        self.target = np.concat((self.target[shuffle[:-games.n]], games.mines))
        self.states = np.concat((self.states[shuffle[:-games.n]], games.game_state()))
        weights = self._compute_weights(games)
        self.weights = np.concat((self.weights[shuffle[:-games.n]], weights))

    def _compute_weights(self, games: Game) -> np.ndarray:
        return (
            games.losing_moves()*(self.losing_moves_weight-1)
            + 1 - games.open_cells - games.flags)

    def __len__(self):
        return self.target.shape[0]

    def __getitem__(self, idx):
        
        sample = self.states[idx]
        target = self.target[idx]
        weights = self.weights[idx]

        if self.transform:
            sample, target, weights = self.transform(sample, target, weights)

        return sample, target, weights
    

class OnHotEncodingTransform:
    def __init__(self, padding: int):
        self.padding = padding

    def __call__(self, numbers: np.ndarray, mines: np.ndarray = None, weights: np.ndarray = None) -> torch.Tensor:
        # numbers can be (h,w) or (n,h,w)
        x = torch.from_numpy(numbers).long() # long required by one_hot function
        y = torch.from_numpy(mines).float() if mines is not None else None
        w = torch.from_numpy(weights).float() if weights is not None else None
        permute, concat_dim = ((2, 0, 1), 0) if len(numbers.shape) == 2 else ((0, 3, 1, 2), 1)
        # 11 classes: numbers 0 to 8, 9 for closed cells, 10 for flags
        x = nn.functional.one_hot(x, 11).permute(*permute) # move the encoding from last to channel dimension
        x = nn.functional.pad(x, [self.padding]*4)
        # padding mask is the 12th channel, indicating cells outside the grid
        padding_mask = 1-nn.functional.pad(torch.ones(numbers.shape), [self.padding]*4)
        x = torch.concat((x, torch.unsqueeze(padding_mask, concat_dim)), dim=concat_dim)
        return x, y, w
    
class PatchMLPModel:
    def __init__(self, patch_radius: int, device: str = 'cpu'):
        self.pad = patch_radius
        self.device = device
        self.transform = OnHotEncodingTransform(patch_radius)
        self.model = PatchMLP(
            in_channels=12,
            out_channels=1,
            patch_size=2*patch_radius+1,
            padding=0,
            layer_units=[200]*4,
            out_activation=nn.Sigmoid()
        )
        self.train_loss_log = []
        self.test_loss_log = []

    def loss(self, pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        squeezed = pred.view(pred.size(0), pred.size(2), pred.size(3))
        return nn.functional.binary_cross_entropy(squeezed, target, weight=weights)
    
    def __call__(self, game_state: np.ndarray) -> np.ndarray:
        self.model.eval()
        self.model.to(self.device)
        x, _, _ = self.transform(game_state)
        x.to(self.device)
        return self.model(x).view(game_state.shape).cpu().detach().numpy()
    
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
    
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
