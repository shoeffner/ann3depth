import tensorflow as tf


class FlatNetwork:

    def __init__(self, input_shape, output_shape):
        self.input = tf.placeholder(tf.float32, shape=(None, ) + input_shape)
        self.target = tf.placeholder(tf.float32, shape=(None, ) + output_shape)
        print(self.input, self.target)
