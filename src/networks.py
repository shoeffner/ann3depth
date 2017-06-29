import itertools
import os

import tensorflow as tf
import numpy as np

import data


class DepthMapNetwork:

    def __init__(self, input_shape, output_shape):
        self.ckpt_path = os.path.join('.', os.environ['CKPT_DIR'],
                                      '{}'.format(self.__class__.__name__))

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.uint8,
                                        shape=(None, ) + input_shape)
            self.target = tf.placeholder(tf.uint8,
                                         shape=(None, ) + output_shape)
            # Grayscale
            cast = tf.cast(self.input, tf.float32) / 255
            gray = tf.image.rgb_to_grayscale(cast)

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
                                        padding='same', activation=tf.nn.relu)
            conv.activation = tf.nn.sigmoid

            self.output = tf.squeeze(conv)

            loss = tf.reduce_sum(
                tf.squared_difference(self.output,
                                      tf.cast(self.target, tf.float32) / 255)
            )
            self.optimizer = tf.train.AdamOptimizer().minimize(loss)

            self.saver = tf.train.Saver()

    def __call__(self, dataset, epochs=100, batchsize=32):
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())

            for epoch in range(1, 1 + epochs):
                print(f'Epoch: {epoch}')

                for b_in, b_out in data.as_matrix_batches(dataset, batchsize):
                    s.run(self.optimizer,
                          {self.input: b_in, self.target: b_out})

                if not epoch % 10:
                    self.saver.save(s, str(self.ckpt_path))

            results = s.run(self.output,
                            {self.input: np.array([d.img for d in dataset]),
                             self.target: np.array([d.depth for d in dataset])})
        print(results)
        for i, result in enumerate(results):
            dataset[i].result = result.squeeze() * 255
