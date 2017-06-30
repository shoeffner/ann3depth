import itertools
import os
import time

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
            i_boundary = min(steps_h, steps_v) // 2 + 2
            for i in range(i_boundary):
                conv = tf.layers.conv2d(conv, 1, 3,
                                        strides=(1 + i % 2, 2 - i % 2),
                                        padding='same',
                                        activation=(tf.nn.relu
                                                    if i != i_boundary - 1 else
                                                    tf.nn.sigmoid)
                                        )
            squeeze = tf.squeeze(conv)

            self.output = squeeze * 255

            loss = tf.reduce_sum(
                tf.squared_difference(self.output,
                                      tf.cast(self.target, tf.float32))
            )
            self.optimizer = tf.train.AdamOptimizer().minimize(loss)

            self.saver = tf.train.Saver()

    def __call__(self, dataset, epochs=100, batchsize=32):
        start = time.time()
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())

            for epoch in range(1, 1 + epochs):
                epoch_start = time.time()
                print(f'Epoch: {epoch}')

                for b_in, b_out in data.as_matrix_batches(dataset, batchsize):
                    s.run(self.optimizer,
                          {self.input: b_in, self.target: b_out})

                print(f'Elapsed time: {time.time() - start}',
                      f'Epoch time: {time.time() - epoch_start}')
                if not epoch % 10:
                    print('Saving')
                    self.saver.save(s, str(self.ckpt_path))

            results = s.run(self.output,
                            {self.input: np.array([d.img for d in dataset]),
                             self.target: np.array([d.depth for d in dataset])})

        for i, result in enumerate(results):
            dataset[i].result = result.squeeze()
