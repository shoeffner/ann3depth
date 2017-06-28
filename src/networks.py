import itertools
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

            # Grayscale
            gray = tf.image.rgb_to_grayscale(self.input)

            # Scale to nearest multiple of output size
            resize = tf.image.resize_images(gray,
                                            tuple(itertools.starmap(
                                                lambda x, y: x // y * y,
                                                zip(input_shape, output_shape))
                                                ))

            # convolve to output size, alternating between horizontal and
            # vertical
            steps_h, steps_v = map(lambda x: x[0] // x[1],
                                   zip(input_shape, output_shape))
            conv = resize
            for i in range(min(steps_h, steps_v) // 2 + 2):
                conv = tf.layers.conv2d(conv, 1, 3,
                                        strides=(1 + i % 2, 2 - i % 2),
                                        padding='same')

            self.output = conv

            self.saver = tf.train.Saver()

    def __call__(self, dataset):
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())

            results = s.run(self.output,
                            {self.input: np.array([d.img for d in dataset])})

            self.saver.save(s, str(self.ckpt_path))

        for i, result in enumerate(results):
            dataset[i].result = result.squeeze()
