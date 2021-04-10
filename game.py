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
        self.scores = np.full(n, self.size-mines, dtype=np.int)
        self.mines_scores = np.full(n, self.mines, dtype=np.int)
        self.active_grids = np.ones(n, dtype=bool)
        self.won = np.zeros(n, dtype=bool)
        self._range = np.arange(n)
        self.last_opened = np.full(n, -1, dtype=np.int)

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
        self.won = self.scores == 0
        self.active_grids[self.won] = False
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
            self.won = self.scores == 0

    def pyplot_games(self, full_grid = False, cmap=None, cols = 2):
        rows = int(np.ceil(self.n/cols))
        f, axs = plt.subplots(rows, cols, figsize=(12, 12*rows/cols))
        if cmap is None:
            color = np.logical_not(self.states)
        else:
            color = cmap
        data = self.grids if full_grid else self.visible_grids
        for i, ax in enumerate(axs.ravel()):
            if i>= self.n:
                f.delaxes(ax)
                continue
            t = data[i].reshape(self.rows, self.columns)
            state = self.states[i].reshape(self.rows, self.columns)
            colors = color[i].reshape(self.rows, self.columns).astype(np.single).copy()
            last = self.last_opened[i]
            if last >= 0:
                #colors[last//self.rows, last%self.columns] = 0.5
                rect = plt.Rectangle((last%self.columns - .5, last//self.rows - .5), 1, 1, fill=False, color="red", linewidth=4)
                ax.add_patch(rect)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks(np.linspace(0.5, self.columns - 1.5, self.columns - 1))
            ax.set_yticks(np.linspace(0.5, self.rows - 1.5, self.rows - 1))
            ax.imshow(colors)
            ax.grid(color="w", linestyle='-', linewidth=1)
            for r in range(self.rows):
                for c in range(self.columns):
                    if state[r, c] > 0:
                        ax.text(c, r, t[r, c], ha="center", va="center", color="w", weight='bold')
                    elif full_grid:
                        s = 'x' if t[r,c] < 0 else t[r,c]
                        ax.text(c, r, s, ha="center", va="center", color="w")
                    elif cmap is not None:
                        ax.text(c, r, "{:.1f}".format(colors[r,c]), ha="center", va="center", color="grey", size='small')
        return f, axs