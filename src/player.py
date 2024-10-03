import numpy as np

from .game import Game

class Player:
    def __init__(self, model):
       self.model = model

    def play(self, game: Game):
        i = 1
        while np.any(game.active_games):
            print(f'step {i}')
            self.step(game)
            i += 1

    def step(self, game):
        prob = self.model.predict(game, active_only=True)
        closed_prob = prob+game.states[game.active_grids]
        cells = np.argmin(closed_prob, axis=1)
        game.open(cells)
        return prob, cells

class ThresholdPlayer(Player):
    def __init__(self, model, open_thresh: int=0.01, flag_thresh: int=0.99):
        super().__init__(model)
        self.open_tresh = open_thresh
        self.flag_tresh = flag_thresh

    def step(self, game: Game):
        p = self.model(game.game_state(active_only=True))
        full_cells = game.open_cells[game.active_games] + game.flags[game.active_games]
        p_for_min = p + full_cells
        p_for_max = p - full_cells
        to_open = p_for_min < self.open_tresh # open cells below open threshold
        to_flag = p_for_max > self.flag_tresh # flag cells above flag threshold
        no_moves = np.logical_not(np.any(to_open, axis=(1,2)) | np.any(to_flag, axis=(1,2))) # games without new open of flags
        p_no_moves = p_for_min[no_moves]
        to_open[no_moves] = p_no_moves <= p_no_moves.min(axis=(1,2), keepdims=True)+0.01 # open the cells with minimum
        no_moves = np.logical_not(np.any(to_open, axis=(1,2)) | np.any(to_flag, axis=(1,2)))
        if np.any(no_moves):
            raise Exception('No moves could be selected') # it there are no moves we go into infinite loop
        game.open_and_flag(to_open, to_flag)
        return p, to_open, to_flag