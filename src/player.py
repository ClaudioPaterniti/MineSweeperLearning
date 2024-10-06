import numpy as np

from .game import Game

class Player:
    def __init__(self, model):
       self.model = model

    def play(self, game: Game, turns: int = np.inf):
        """
        :param turns: how many turns to play, default: np.inf = until all games are lost or won
        """
        turns = np.inf if turns < 0 else turns
        i = 0
        while i < turns and np.any(game.active_games):
            self.step(game)
            i += 1

    def step(self, game: Game):
        raise NotImplementedError()

class ThresholdPlayer(Player):
    def __init__(self, model, open_thresh: int=0.01, flag_thresh: int=0.99):
        super().__init__(model)
        self.open_tresh = open_thresh
        self.flag_tresh = flag_thresh

    def step(self, game: Game):
        if not np.any(game.active_games):
            print('no active games')
            return None, None, None
        p = self.model(game.game_state(active_only=True), game.mines_n[game.active_games])
        filled = game.open_cells[game.active_games] + game.flags[game.active_games]
        p_for_min = p + filled # set already filled cells > 1 to get meaningful low probabilties
        p_for_max = p - filled # set already filled cells < 0 to get meaningful high probabilities
        to_open = p_for_min < self.open_tresh # open cells below open threshold
        to_flag = p_for_max > self.flag_tresh # flag cells above flag threshold
        no_moves = np.logical_not(np.any(to_open, axis=(1,2)) | np.any(to_flag, axis=(1,2))) # games without new open no flags
        if np.any(no_moves):
            idxs = p_for_min.reshape(p.shape[0], -1)[no_moves].argmin(axis=1)
            h_ids = idxs//game.columns
            w_ids = idxs%game.columns
            to_open[no_moves, h_ids, w_ids] = 1 # open the cells with minimum
        game.open_and_flag(to_open, to_flag)
        return p, to_open, to_flag