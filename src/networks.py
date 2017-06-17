import tensorflow as tf
import numpy as np


class FlatNetwork:

    def __init__(self, input_shape, output_shape):
        self.input = tf.placeholder(tf.float32,
                                    shape=(None, ) + input_shape)
        self.convlayer = tf.layers.conv2d(self.input, 1, 1)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def __call__(self, dataset):
        with self.session as s:
            results = s.run(self.convlayer,
                            {self.input: np.array([d.img for d in dataset])})
        self.session.close()
        for i, result in enumerate(results):
            dataset[i].result = result.squeeze()
