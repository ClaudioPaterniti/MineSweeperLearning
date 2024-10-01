import numpy as np

from matplotlib.colors import Normalize as ColorNormalize
import matplotlib.pyplot as plt

from . import utils

class Game:
    def __init__(self, rows: int=9, columns: int=9, mines_n:int =10, n: int=1):
        """:param n: number of parallel games"""
        self.n = n
        self.rows = rows
        self.columns = columns
        self.size = rows*columns
        self.mines_n = mines_n
        self._range = np.arange(n)
        self.mines = utils.random_binary_matrices((n, rows, columns), mines_n)
        self.numbers = self._compute_number_cells() # grids with number of neighbor mines
        self.open_cells = np.zeros_like(self.mines)
        self.closed_n = np.full(n, self.size, dtype=int)
        self.flags = np.zeros_like(self.mines)
        self.active_games = np.ones(n, dtype=bool)
        self.won = np.zeros(n, dtype=bool)
        self.last_opened = np.zeros_like(self.mines)
        self.last_flagged = np.zeros_like(self.mines)

    def _compute_number_cells(self): # compute the minesweeper numbers from the mine placements
        grids = np.zeros_like(self.mines)
        pad = np.pad(self.mines, [(0,0), (1,1), (1,1)])
        for i in range(-1,2):
            for j in range(-1,2):
                grids += pad[:, 1+i:self.rows+1+i , 1+j:self.columns+1+j] # the number of a cell is the sum of the nighbour mines
        grids[self.mines.astype(bool)] = -1
        return grids
    
    def game_state(self, active_only: bool = True):
        """return the games with:
        0-8: open cell with corresponding minesweeper number,
        9: closed cell,
        10: flag"""
        state = self.numbers*self.open_cells + 9*(1-self.open_cells) + self.flags
        return state[self.active_games] if active_only else state
    
    def open(self, cells: np.ndarray) -> np.ndarray[bool]:
        """Open cells in active games, return bool array (n) where False means a mine has been open
        
        :param cells: binary matrices (n,h,w)"""
        correct = np.logical_not(np.any(self.mines[self.active_games]*cells, axis=(1,2)))
        self.last_opened = cells
        self.open_cells = np.bitwise_or(self.open_cells, cells*(1-self.mines))
        self.closed_n = self.size - self.open_cells.sum(axis=2).sum(axis=1)
        self.active_games[self.active_games] = correct
        self.won =  self.closed_n == self.mines_n
        self.active_games[self.won] = False
        return correct    
    
    def flag(self, flags: np.array) -> np.ndarray[bool]: # wrong flag make you lose for training convenience
        """Mark active games cells as flagged, return bool array (n) where False means a wrong flag has been placed

        :param flags: binary ndarray (n,h,w) of the cells to flag
        """
        flags = flags*(1-self.open_cells[self.active_games])
        self.flags[self.active_games] = np.bitwise_or(self.flags[self.active_games], flags)
        correct = np.logical_not(np.any((1-self.mines[self.active_games])*flags, axis=(1,2)))
        self.active_games[self.active_games] = correct
        self.last_flagged = flags
        return correct
    
    def random_open(self, rate: float):
        """Open random cells"""
        to_open = utils.random_binary_matrices((self.n, self.rows, self.columns), int(rate*self.size))
        to_open *= 1-self.mines # do not open mines
        self.open_cells = to_open
        self.closed_n = self.size - to_open.sum(axis=2).sum(axis=1)

    def random_flags(self, rate: float):
        """Flag random mines"""
        to_flag = utils.random_binary_matrices((self.n, self.rows, self.columns), int(rate*self.size))
        to_flag *= self.mines # only flag mines
        self.flags = to_flag

    def losing_moves(self) -> np.ndarray:
        """return (n,h,w) binary array of wrong openings or flags in the last actions"""
        return self.last_opened*self.mines + self.last_flagged*(1-self.mines)

    def pyplot_game(self, idx: int, full_grid: bool = False, mine_probs: np.ndarray=None, cmap = plt.cm.jet):
        """plot game state
        :param idx: game index
        :param full_grid: whether to print the full grid or only the visible part
        :param mine_probs: (h,w) ndarray of mine probabilities to plot
        """
        def style(x: int, p: float = None) -> dict:
            if x < 0: return {'s': 'x',  'weight': 'bold'}
            if x < 9: return {'s': x,  'weight': 'bold'}
            if x == 9: return {'s': '{:.1f}'.format(p) if p else ''}
            if x == 10: return {'s': '?',  'weight': 'bold'}

        state = self.numbers[idx] if full_grid else self.game_state()[idx]
        color = (mine_probs+0.2)*(1-self.open_cells[idx]) if mine_probs is not None\
              else self.open_cells[idx]*0.2+self.flags[idx]
        plt.matshow(color, cmap=cmap, norm=ColorNormalize(vmin=0, vmax=1))
        ax = plt.gca()
        ax.grid(color="w", linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(self.columns)-0.5)
        ax.set_yticks(np.arange(self.rows)-0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for r in range(self.rows):
            for c in range(self.columns):
                    v, p = state[r, c], mine_probs[r, c] if mine_probs is not None else None
                    ax.text(c, r, ha="center", va="center", color="w", **style(v, p))
        plt.show()
        return ax