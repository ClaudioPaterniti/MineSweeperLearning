import numpy as np
import matplotlib.pyplot as plt

class Game:
    def __init__(self, rows=9, columns=9, mines=10, n=1):
        self.n = n
        self.rows = rows
        self.columns = columns
        self.size = rows*columns
        self.mines = mines
        self.fields = self._compute_fields()
        self.grids = self._compute_grids()
        self.states = np.zeros((n, self.size), dtype=np.int)
        self.visible_grids = self.states.copy()
        self.scores = np.ones(n, dtype=np.int)*(self.size-mines)
        self.active_grids = np.ones(n, dtype=bool)
        self._range = np.arange(n)
        self.last_opened = -np.ones(n, dtype=np.int)

    def _compute_fields(self):
        fields = np.zeros((self.n,self.size), dtype=np.int)
        for i in range(self.n):
            fields[i, np.random.choice(self.size, self.mines, replace=False)] = 1
        return fields

    def _compute_grids(self):
        pad = np.pad(self.fields.reshape(self.n, self.rows, self.columns), [(0,0), (1,1), (1,1)])
        grids = np.zeros((self.n, self.rows,self.columns), dtype=np.int)
        for i in range(-1,2):
            for j in range(-1,2):
                grids += pad[:, 1+i:self.rows+1+i , 1+j:self.columns+1+j]
        grids = grids.reshape(self.n, self.size)
        grids[self.fields.astype(bool)] = -1
        return grids

    def open(self, c):
        opened = (self._range[self.active_grids],c)
        self.last_opened[self.active_grids] = c
        self.visible_grids[opened] = self.grids[opened]
        non_bombs = self.grids[opened]!=-1
        self.active_grids[opened[0]] = non_bombs
        self.states[opened[0][non_bombs], opened[1][non_bombs]] = 1
        self.scores[self.active_grids] -= 1
        self.active_grids[self.scores == 0] = False
        return self.grids[opened]

    def open_zero(self, c=None, pad_grid=None, pad_state=None):
        first_call=False
        if c is None:
            c = np.argmin(np.absolute(self.grids), axis=1)
            zeros = self.grids[self._range, c]==0
            c = (self._range[zeros], c[zeros])
        if pad_grid is None:
            first_call = True
            pad_grid = np.pad(self.grids.reshape(self.n, self.rows, self.columns),
                              [(0,0), (1,1), (1,1)],  constant_values=-1)
            pad_state = np.pad(self.states.reshape(self.n, self.rows, self.columns),
                               [(0, 0), (1, 1), (1, 1)],  constant_values=-1)
            c = (c[0], np.floor_divide(c[1], self.rows)+1, c[1]%self.columns+1)
        for i in range(-1, 2):
            for j in range(-1, 2):
                t = (c[0], c[1]+i, c[2]+j)
                to_open = pad_state[t]==0
                t = (t[0][to_open], t[1][to_open], t[2][to_open])
                pad_state[t] = 1
                self.scores[t[0]] -= 1
                zeros = pad_grid[t] == 0
                if np.any(zeros):
                    t = (t[0][zeros], t[1][zeros], t[2][zeros])
                    self.open_zero(t, pad_grid, pad_state)
        if first_call:
            self.states = pad_state[:,1:-1,1:-1].reshape(self.n, self.size)
            self.visible_grids = self.grids*self.states
            self.active_grids = self.scores > 0

    def pyplot_games(self, full_grid = False, map=None):
        f, axs = plt.subplots(2, int(np.ceil(self.n/2)), figsize=(12, 3*self.n))
        if map is None:
            map = np.logical_not(self.states)
        data = self.grids if full_grid else self.visible_grids
        for i, ax in enumerate(axs.ravel()[:self.n]):
            t = data[i].reshape(self.rows, self.columns)
            state = self.states[i].reshape(self.rows, self.columns)
            colors = map[i].reshape(self.rows, self.columns).astype(np.single).copy()
            last = self.last_opened[i]
            if last >= 0:
                colors[last//self.rows, last%self.columns] = 0.5
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(0.5, self.columns - 1.5, self.columns - 1))
            ax.set_yticks(np.linspace(0.5, self.rows - 1.5, self.rows - 1))
            ax.imshow(colors)
            ax.grid(color="w", linestyle='-', linewidth=1)
            for i in range(self.rows):
                for j in range(self.columns):
                    if state[i, j] > 0:
                        ax.text(j, i, t[i, j], ha="center", va="center", color="w")
                    elif full_grid:
                        s = 'x' if t[i, j] < 0 else t[i,j]
                        ax.text(j, i, s, ha="center", va="center", color="w")
                        
        return f, axs