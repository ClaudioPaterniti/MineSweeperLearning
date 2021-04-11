import pickle
import os

import tensorflow as tf
import numpy as np

class Net:

    def _process_input(self, game, mask):
        raise NotImplementedError


    def fit(self, games, lr_scheduler, cp_path, **train_args):
        x = []; y = []
        for game in games:
            _x, _y = self._process_input(game, np.ones(game.n, dtype=bool))
            x.append(_x); y.append(_y)
        x = np.concatenate(x); y = np.concatenate(y)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_path, 
            verbose=1,
            save_weights_only=True)
        lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler, verbose=1)
        self.model.fit(x, y, callbacks=[cp_callback, lr_callback], **train_args)

    def predict(self, game, active_only = False):
        mask = game.active_grids if active_only else np.ones(game.n, dtype=bool)
        x, _ = self._process_input(game, mask)
        return self.model.predict(x)


    def save(self, path, name):
        if not os.path.exists(path):
            os.makedirs(path)
        temp = self.model
        self.model = None
        with open(os.path.join(path, name + '.pkl'), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        temp.save(os.path.join(path,name+ '_tfmodel', ))
        self.model = temp

    @staticmethod
    def load(path, name):
        with open(os.path.join(path, name + '.pkl'), 'rb') as file:
            net = pickle.load(file)
        net.model = tf.keras.models.load_model(os.path.join(path,name+ '_tfmodel', ))
        return net

class Minesweeper_dense_net(Net):
    def __init__(self, output_size, layout):
        self.output_size = output_size
        self.layout = layout
        self.model = tf.keras.Sequential()
        for i, units in enumerate(self.layout):
            self.model.add(tf.keras.layers.Dense(units, activation='sigmoid', name=str(i)))
        self.model.add(tf.keras.layers.Dense(self.output_size, activation='sigmoid', name='output'))
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    def _process_input(self, game, mask):
        x = np.concatenate((game.visible_grids[mask], game.states[mask]), axis=1)
        y = game.fields[mask]
        return x,y

class Minesweeper_single_cell_net(Minesweeper_dense_net):
    def __init__(self, window_radius, layout, mines_feature = False):
        super().__init__(1, layout)
        self.w = window_radius
        self.mines_feature = mines_feature

    def _extract_squares(self, grids, n, x, y, cells, padc = 0):
        o = grids.reshape((n, x, y))
        o = np.pad(o, [(0, 0), (self.w, self.w), (self.w, self.w)], constant_values=padc)
        o = o.reshape((n, (x + 2*self.w)*(y + 2*self.w)))
        rows, cols = np.nonzero(cells)
        side = 2*self.w + 1
        _x = x + 2*self.w
        cols += (_x + 1)*self.w + (cols//x)*2*self.w
        square = np.concatenate([np.arange(side) + i*_x for i in range(side)])
        rem_center = np.ones(side**2, dtype=int)
        rem_center[side**2//2] = 0
        square = square[rem_center.astype(bool)] - (_x + 1)*self.w
        cols = square + cols.reshape((-1, 1))
        rows = rows.reshape((-1, 1))
        return o[rows, cols]

    def _process_input(self, game, mask):
        n, x, y, cells = np.count_nonzero(mask), game.columns, game.rows, np.logical_not(game.states[mask])
        values = self._extract_squares(game.visible_grids[mask], n, x, y, cells)
        states = self._extract_squares(game.states[mask], n, x, y, cells)
        inside = self._extract_squares(np.ones_like(cells, dtype=np.int), n, x, y, cells)
        score_rep = game.scores[mask]+game.mines
        scores = np.repeat(score_rep, score_rep).reshape((-1,1))
        x = np.concatenate((values, states, inside, scores), axis=1)
        if self.mines_feature:
            mine_scores = np.repeat(game.mines_scores[mask], score_rep).reshape((-1,1))
            x = np.concatenate((x, mine_scores), axis=1)
        y = game.fields[mask][cells]
        return x,y

    def predict(self, game, active_only=False):
        mask = game.active_grids if active_only else np.ones(game.n, dtype=bool)
        x, _ = self._process_input(game, mask)
        pred = self.model.predict(x)
        p = np.zeros_like(game.grids[mask], dtype=np.float)
        p[np.logical_not(game.states[mask])] = pred.flatten()
        return p
