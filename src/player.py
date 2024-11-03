import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation

from .game import Game
from .models.patch_mlp import MinesweeperModel
from .utils import pyplot_game, vanishing_colormap

class Player:

    def play(self, game: Game, turns: int = np.inf) -> int:
        """
        :param turns: how many turns to play, default: np.inf = until all games are lost or won
        """
        turns = np.inf if turns < 0 else turns
        i = 0
        while i < turns and np.any(game.active_games):
            self.step(game)
            i += 1
        return i

    def step(self, game: Game) -> tuple[np.ndarray, np.ndarray]:
        """Play one turn.
        Returns two binary (n,h,w) arrays with respectively the open and flagged cells"""
        if not np.any(game.active_games):
            print('no active games')
            return None, None
        to_open, to_flag = self.get_moves(game.game_state(True), game.mines_n[game.active_games])
        game.move(to_open, to_flag)
        return to_open, to_flag

    def get_game_moves(self, game: Game) -> tuple[np.ndarray, np.ndarray]:
        to_open, to_flag = self.get_moves(game.game_state(), game.mines_n)
        return to_open, to_flag

    def get_moves(self, state: np.ndarray, tot_mines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        :param state: (n,w,h) with games state
        :param tot_mines: (n) with tot number of mines in the game
        Returns two binary (n,h,w) arrays with respectively the open and flagged cells"""
        raise NotImplementedError()

    def plot_game_moves(self, game: Game, idx: int = 0, **plot_kwargs):
        to_open, to_flag = self.get_game_moves(game[idx])
        game.pyplot_game(idx, highlighted=to_flag[0] - to_open[0], **plot_kwargs)

    def plot_moves(self, state: np.ndarray, tot_mines: int, **plot_kwargs):
        """
        :param state: (w,h) single game state
        :param tot_mines: int with tot number of mines in the game"""
        to_open, to_flag = self.get_moves(np.expand_dims(state, 0), np.expand_dims(tot_mines, 0))
        pyplot_game(state, highlighted=to_flag[0] - to_open[0], **plot_kwargs)

class ThresholdPlayer(Player):
    def __init__(self, model: MinesweeperModel, open_thresh: int=0.01, flag_thresh: int=0.99):
        super().__init__()
        self.model = model
        self.open_tresh = open_thresh
        self.flag_tresh = flag_thresh

    def get_moves(self, state: np.ndarray, tot_mines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = self.model(state, tot_mines)
        known = (state != 9).view(np.int8)
        p_for_min = p + known # set known cells > 1 to get meaningful low probabilties
        p_for_max = p - known # set known cells < 0 to get meaningful high probabilities
        to_open = p_for_min < self.open_tresh # open cells below open threshold
        to_flag = p_for_max > self.flag_tresh # flag cells above flag threshold
        no_moves = np.logical_not(np.any(to_open, axis=(1,2)) | np.any(to_flag, axis=(1,2))) # games without new open or flags
        if np.any(no_moves):
            idxs = p_for_min.reshape(p.shape[0], -1)[no_moves].argmin(axis=1)
            h_ids = idxs//state.shape[2]
            w_ids = idxs%state.shape[2]
            to_open[no_moves, h_ids, w_ids] = 1 # open the cell with minimum
        return to_open.view(np.int8), to_flag.view(np.int8)

class GameAnimation():
    def __init__(self, game: Game, player: Player, interval=1500, repeat=False, cell_size: int = 0.4):
        self.player = player
        self.game = game

        self.fig, self.ax = plt.subplots(figsize=(self.game.columns*cell_size, self.game.rows*cell_size))
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.ax, self.s, self.h = pyplot_game(
            self.game.game_state()[0], highlighted=self.game.open_cells[0], print_zeros=False, ax=self.ax)
        self.animation = matplotlib.animation.FuncAnimation(
            fig=self.fig, func=self._update, frames=self._frames,
            interval=interval, repeat=repeat, save_count = 100, blit = True)

    def _frames(self):
        while np.any(self.game.active_games):
            yield 1

    def _update(self, frame):
        self.player.step(self.game)
        state = self.game.game_state()[0]
        highlighted = self.game.last_flagged[0] - self.game.last_opened[0]
        pyplot_game(state, highlighted=highlighted, init=False, ax=self.ax, state_artist=self.s, hghl_artist=self.h)
        return [self.s, self.h]