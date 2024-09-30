import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm

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
        self.mines = utils.random_binary_matrices((rows, columns), mines_n, n)
        self.numbers = self._compute_number_cells() # grids with number of neighbor mines
        self.open_cells = np.zeros_like(self.mines)
        self.closed_n = np.full(n, self.size, dtype=int)
        self.flags = np.zeros_like(self.mines)
        self.active_games = np.ones(n, dtype=bool)
        self.won = np.zeros(n, dtype=bool)
        self.last_opened = np.full((n,2), -1, dtype=int)    

    def _compute_number_cells(self): # compute the minesweeper numbers from the mine placements
        grids = np.zeros_like(self.mines)
        pad = np.pad(self.mines, [(0,0), (1,1), (1,1)])
        for i in range(-1,2):
            for j in range(-1,2):
                grids += pad[:, 1+i:self.rows+1+i , 1+j:self.columns+1+j] # the number of a cell is the sum of the nighbour mines
        grids[self.mines.astype(bool)] = -1
        return grids
    
    def visible_numbers(self):
        """return the open cells numbers, closed cells are set to -1 to distinguish them from zeroes"""
        return (self.numbers + 1)*self.open_cells - 1

    def open(self, indices: np.ndarray):
        """Open on cell per active game, returns the content of the opened cells
        
        :param indices: array (2,n) with the coordinates of the cell to open"""
        opened = (self._range[self.active_games], indices[0], indices[1])
        self.last_opened[self.active_games] = indices.transpose()
        not_losing = np.logical_not(self.mines[opened].astype(bool))
        self.active_games[opened[0]] = not_losing
        self.open_cells[opened] = 1
        self.closed_n -= 1
        self.won =  self.closed_n == self.mines_n
        self.active_games[self.won] = False
        return self.numbers[opened]
    
    def random_open(self, rate: float):
        """Open random cells"""
        to_open = utils.random_binary_matrices((self.rows, self.columns), int(rate*self.size), self.n)
        to_open *= 1-self.mines # do not open mines
        self.open_cells = to_open
        self.closed_n = self.size - to_open.sum(axis=2).sum(axis=1)

    def random_flags(self, rate: float):
        """Flag random mines"""
        to_flag = utils.random_binary_matrices((self.rows, self.columns), int(rate*self.size), self.n)
        to_flag *= self.mines # only flag mines
        self.flags = to_flag
    
    def flag(self, flags: np.array):
        """Mark active games cells as flagged

        :param flags: binary ndarray (n,h,w) of the cells to flag
        """
        self.flags[self.active_games] += flags
        self.flags.clip(0, 1)

    def pyplot_games(self, full_grid: bool = False, mine_probs: np.ndarray=None, cols: int = 2):
        """plot games state

        :param full_grid: whether to print the full grid or only the visible part
        :param mine_probs: nxhxw ndarray of mine probabilities to plot
        :param cols: plot columns number
        """
        rows = int(np.ceil(self.n/cols))
        visible_numbers = self.visible_numbers()
        f, axs = plt.subplots(rows, cols, figsize=(18, 14*rows/cols))
        if rows*cols == 1:
            axs = np.array([axs])
        color =  mine_probs if mine_probs is not None else 0.5 - self.open_cells/2 + self.flags
        data = self.numbers if full_grid else visible_numbers
        for i, ax in enumerate(axs.ravel()):
            if i>= self.n:
                f.delaxes(ax)
                continue
            t = data[i]
            state = self.open_cells[i]
            flag = self.flags[i]
            colors = color[i].copy()+0.2 # to distinguish open cells
            colors *= 1-self.open_cells[i]
            last = self.last_opened[i]
            if last[0] >= 0:
                rect = plt.Rectangle((last[1] - .5, last[0] - .5), 1, 1, fill=False, color="red", linewidth=4)
                ax.add_patch(rect)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(0.5, self.columns - 1.5, self.columns - 1))
            ax.set_yticks(np.linspace(0.5, self.rows - 1.5, self.rows - 1))
            cmap = matplotlib.cm.jet
            cmap.set_bad(color='red')
            ax.imshow(colors, cmap=cmap)
            ax.grid(color="w", linestyle='-', linewidth=1)
            for r in range(self.rows):
                for c in range(self.columns):
                    if state[r, c] > 0:
                        s = 'x' if t[r, c] < 0 else t[r, c]
                        ax.text(c, r, s, ha="center", va="center", color="w", weight='bold')
                    elif flag[r, c] > 0:
                        s = '?' 
                        ax.text(c, r, s, ha="center", va="center", color="w", weight='bold')
                    elif full_grid:
                        s = 'x' if t[r,c] < 0 else t[r,c]
                        ax.text(c, r, s, ha="center", va="center", color="w")
                    elif mine_probs is not None:
                        ax.text(c, r, "{:.1f}".format(colors[r,c]-0.2), ha="center", va="center", color="grey", size='small')
        return f, axs