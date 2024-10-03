import numpy as np

from matplotlib.colors import Normalize as ColorNormalize
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from . import utils

class Game:
    def __init__(self, rows: int=9, columns: int=9, mines_n:int =10, n: int=1):
        """:param n: number of parallel games"""
        self.n = n
        self.rows = rows
        self.columns = columns
        self.size = rows*columns
        self.mines_n = mines_n
        self.mines = utils.random_binary_matrices((n, rows, columns), mines_n)
        self.numbers = self._compute_number_cells() # grids with number of neighbor mines
        self.open_cells = np.zeros_like(self.mines)
        self.flags = np.zeros_like(self.mines)
        self.active_games = np.ones(n, dtype=bool)
        self.won = np.zeros(n, dtype=bool)
        self.last_opened = np.zeros_like(self.mines)
        self.last_flagged = np.zeros_like(self.mines)

    def __getitem__(self, key):
        g = Game(self.rows, self.columns, self.mines_n, 0)
        if isinstance(key, int):
            key = slice(key, key+1)
        g.mines = self.mines[key]
        g.n = g.mines.shape[0]
        g.numbers = self.numbers[key]
        g.open_cells = self.open_cells[key]
        g.flags = self.flags[key]
        g.active_games = self.active_games[key]
        g.won = self.won[key]
        g.last_opened = self.last_opened[key]
        g.last_flagged = self.last_flagged[key]
        return g



    def _compute_number_cells(self): # compute the minesweeper numbers from the mine placements
        grids = np.zeros_like(self.mines)
        pad = np.pad(self.mines, [(0,0), (1,1), (1,1)])
        for i in range(-1,2):
            for j in range(-1,2):
                grids += pad[:, 1+i:self.rows+1+i , 1+j:self.columns+1+j] # the number of a cell is the sum of the nighbour mines
        grids[self.mines.astype(bool)] = -1
        return grids
    
    def game_state(self, active_only: bool = False):
        """return the games with:
        0-8: open cell with corresponding minesweeper number,
        9: closed cell,
        10: flag"""
        state = self.numbers*self.open_cells + 9*(1-self.open_cells) + self.flags
        return state[self.active_games] if active_only else state
    
    def open(self, cells: np.ndarray, _no_losing: bool = False) -> np.ndarray[bool]:
        """Open cells in active games, return bool array (n) where False means a mine has been open
        
        :param cells: binary matrices (n,h,w)"""
        correct = np.logical_not(np.any(self.mines[self.active_games]*cells, axis=(1,2)))
        self.last_opened[self.active_games] = cells
        self.open_cells[self.active_games] = np.bitwise_or(
            self.open_cells[self.active_games], cells*(1-self.mines[self.active_games]))
        self.won[self.active_games] =  correct*np.all(
            self.open_cells[self.active_games]+self.mines[self.active_games], axis=(1,2))
        if not _no_losing:
            self.active_games[self.active_games] = correct
            self.active_games[self.won] = False
        return correct    
    
    def flag(self, flags: np.ndarray, _no_losing: bool = False) -> np.ndarray[bool]: # wrong flag make you lose for training convenience
        """Mark active games cells as flagged, return bool array (n) where False means a wrong flag has been placed

        :param flags: binary ndarray (n,h,w) of the cells to flag
        """
        flags = flags*(1-self.open_cells[self.active_games])
        self.last_flagged[self.active_games] = flags
        self.flags[self.active_games] = np.bitwise_or(self.flags[self.active_games], flags)
        correct = np.logical_not(np.any((1-self.mines[self.active_games])*flags, axis=(1,2)))
        if not _no_losing: self.active_games[self.active_games] = correct
        return correct
    
    def open_and_flag(self, to_open: np.ndarray, to_flag: np.ndarray) -> np.ndarray[bool]:
        """To open and flag without altering the active games states inbetween"""
        correct_open = self.open(to_open, True)
        correct_flag = self.flag(to_flag, True)
        self.active_games[self.active_games] = correct_open & correct_flag
        self.active_games[self.won] = False
    
    def random_open(self, rate: float):
        """Open random cells"""
        to_open = utils.random_binary_matrices((self.n, self.rows, self.columns), int(rate*self.size))[self.active_games]
        to_open *= 1-self.mines # do not open mines
        self.open(to_open)

    def random_flags(self, rate: float):
        """Flag random mines"""
        to_flag = utils.random_binary_matrices((self.n, self.rows, self.columns), int(rate*self.size))
        to_flag *= self.mines # only flag mines
        self.flag(to_flag)

    def losing_moves(self) -> np.ndarray:
        """return (n,h,w) binary array of wrong openings or flags in the last actions"""
        return self.last_opened*self.mines + self.last_flagged*(1-self.mines)

    def pyplot_game(self,
            idx: int, full_grid: bool = False, mine_probs: np.ndarray=None,
            hightlight_losing_only: bool = False, cmap = plt.cm.viridis) -> Axes:
        """plot game state
        :param idx: game index
        :param full_grid: whether to print the full grid or only the visible part
        :param mine_probs: (h,w) ndarray of mine probabilities to plot,
        :param hightlight_losing_only if true highlights only losing moves, otherwise highlights all last moves
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
        for r in range(self.rows):
            for c in range(self.columns):
                    v, p = state[r, c], mine_probs[r, c] if mine_probs is not None else None
                    ax.text(c, r, ha="center", va="center", color="w", **style(v, p))

        highlights = self.losing_moves()[idx] if hightlight_losing_only else self.last_opened[idx] + self.last_flagged[idx]
        ax.matshow(highlights, cmap=utils.vanishing_colormap(plt.cm.Reds))

        ax.grid(color="w", linestyle='-', linewidth=1)
        ax.set_xticks(np.arange(self.columns)-0.5)
        ax.set_yticks(np.arange(self.rows)-0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.show()
        return ax