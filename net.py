import pickle
import os

import tensorflow as tf
import numpy as np

class Net:

    def _process_input(self, game, active_only):
        raise NotImplementedError

    def fit(self, game, cp_path='../model', cp_rate=100, **train_args):
        STEPS_PER_EPOCH = game.n / train_args['batch_size']
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=cp_path, 
            verbose=1,
            save_weights_only=True,
            save_freq= int(cp_rate * STEPS_PER_EPOCH))
        x,y = self._process_input(game)
        self.model.fit(x, y, callbacks=[cp_callback], **train_args)

    def predict(self, game, active_only = False):
        x,_ = self._process_input(game, active_only)
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
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

    def _process_input(self, game, active_only=False):
        to_process = game.active_grids if active_only else np.ones(game.n, dtype=bool)
        x = np.concatenate((game.visible_grids[to_process], game.states[to_process]), axis=1)
        y = game.fields[to_process]
        return x,y

class Minesweeper_single_cell_net(Minesweeper_dense_net):
    def __init__(self, size, layout):
        super().__init__(1, layout)
        self.size = size

    def _extract_squares(self, grids, n, x, y, padc, cells):
        o = grids.reshape((n, x, y))
        o = np.pad(o, [(0, 0), (self.size, self.size), (self.size, self.size)], constant_values=padc)
        o = o.reshape((n, (x + 2*self.size)*(y + 2*self.size)))
        rows, cols = np.nonzero(cells)
        side = 2*self.size + 1
        _x = x + 2*self.size
        cols += (_x + 1)*self.size + (cols//x)*2*self.size
        square = np.concatenate([np.arange(side) + i*_x for i in range(side)])
        rem_center = np.ones(side**2, dtype=int)
        rem_center[side**2//2] = 0
        square = square[rem_center.astype(bool)] - (_x + 1)*self.size
        cols = square + cols.reshape((-1, 1))
        rows = rows.reshape((-1, 1))
        return o[rows, cols]

    def _process_input(self, game, active_only=False):
        to_process = game.active_grids if active_only else np.ones(game.n, dtype=bool)