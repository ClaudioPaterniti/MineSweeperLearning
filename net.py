import pickle
import os

import tensorflow as tf
import numpy as np

class Net:

    def _process_input(self, game, active_only):
        raise NotImplementedError

    def fit(self, game, **train_args):
        x,y = self._process_input(game)
        self.model.fit(x, y, **train_args)

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
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam())

    def _process_input(self, game, active_only=False):
        to_process = game.active_grids if active_only else np.ones(game.n, dtype=bool)
        x = np.concatenate((game.visible_grids[to_process], game.states[to_process]), axis=1)
        y = game.fields[to_process]
        return x,y

