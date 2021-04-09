import numpy as np

class Player:
    def __init__(self, model, game):
       self.model = model
       self.game = game

    def play(self):
        while np.any(self.game.active_grids):
            self.step()

    def step(self):
        prob = self.model.predict(self.game, active_only=True)
        closed_prob = prob+np.logical_not(self.game.states[self.game.active_grids])
        cell = np.argmin(closed_prob, axis=1)
        self.game.open(cell)
        return prob, cell