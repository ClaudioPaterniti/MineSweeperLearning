import tensorflow as tf

class Minesweeper_dense_net:
    def __init__(self, output_size, layout):
        self.output_size = output_size
        self.layout = layout
        self.model = tf.keras.Sequential()
        for i, units in enumerate(self.layout):
            self.model.add(tf.keras.layers.Dense(units, activation='sigmoid', name=str(i)))
        self.model.add(tf.keras.layers.Dense(self.output_size, activation='sigmoid', name='output'))
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam())

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)
