import os

import tensorflow as tf
import numpy as np


class DepthMapNetwork:

    def __init__(self, input_shape, output_shape):
        self.ckpt_path = os.path.join('.', os.environ['CKPT_DIR'],
                                      '{}'.format(self.__class__.__name__))

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32,
                                        shape=(None, ) + input_shape)
            self.output = tf.layers.conv2d(self.input, 1, 1)
            self.saver = tf.train.Saver()

    def __call__(self, dataset):
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())

            results = s.run(self.output,
                            {self.input: np.array([d.img for d in dataset])})
            self.saver.save(s, str(self.ckpt_path))
        for i, result in enumerate(results):
            dataset[i].result = result.squeeze()
