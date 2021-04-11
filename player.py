import numpy as np

class Player:
    def __init__(self, model):
       self.model = model

    def play(self, game):
        while np.any(game.active_grids):
            self.step(game)

    def step(self, game):
        prob = self.model.predict(game, active_only=True)
        closed_prob = prob+game.states[game.active_grids]
        cells = np.argmin(closed_prob, axis=1)
        game.open(cells)
        return prob, cells

class threshold_policy:
    def __init__(self, threshold=0.9):
        self.c = threshold

    def __call__(self, m, M, scores, mines):
        return M>self.c

class Flags_player(Player):
    def __init__(self, model, policy=threshold_policy()):
        super().__init__(model)
        self.policy = policy

    def step(self, game):
        prob = self.model.predict(game, active_only=True)
        active = game.active_grids
        M = np.argmax(prob, axis=1)
        closed_prob = prob + game.states[active]
        m = np.argmin(closed_prob, axis=1)
        _r = np.arange(len(prob))
        flags = self.policy(prob[_r,m], prob[_r,M], game.scores[active], game.mines_scores[active])
        cells = M*flags + m*np.logical_not(flags)
        game.open(cells, flags)
        return prob, cells

