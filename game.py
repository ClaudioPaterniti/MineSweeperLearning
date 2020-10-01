import numpy as np

class Game:
    def __init__(self, rows=5, columns=5, mines=6):
        self.rows = rows
        self.columns = columns
        self.size = rows*columns
        self.mines = mines
        self.initialize_field()

    def initialize_field(self):
        self.field = np.zeros(self.size, dtype=np.int)
        self.field[np.random.choice(self.size, self.mines, replace=False)] = 1
        self.field = self.field.reshape(self.rows, self.columns)
        self.state = np.ones((self.rows, self.columns), dtype=np.int)
        self.grid = self._compute_grid()
        self.visible_grid = -1*self.state

    def _compute_grid(self):
        pad = np.pad(self.field, 1)
        grid = np.zeros((self.rows,self.columns), dtype=np.int)
        for i in range(-1,2):
            for j in range(-1,2):
                grid += pad[1+i:self.rows+1+i , 1+j:self.columns+1+j]
        grid[self.field.astype(bool)] = -1
        return grid

    def open(self, c):
        self.state[c] = 0
        self.visible_grid[c] = self.grid[c]
        if self.field[c] == 1:
            return True
        return False

    def open_zero(self, c=None):
        if c is None:
            c = np.argwhere(self.grid == 0)
            if c.size == 0:
                return False
            c = tuple(c[0])
            self.state[c] = 0
            self.visible_grid[c] = self.grid[c]
        pad = np.pad(self.state,1)
        c = (c[0]+1,c[1]+1)
        for i in range(-1,2):
            for j in range(-1,2):
                t = (c[0]+i,c[1]+j)
                if pad[t] == 1:
                    t = (t[0] - 1, t[1] - 1)
                    self.state[t] = 0
                    self.visible_grid[t] = self.grid[t]
                    if self.grid[t] == 0:
                        self.open_zero(t)
        return True