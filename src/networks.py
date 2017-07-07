import itertools
import os
import time

from datetime import datetime

import tensorflow as tf
import numpy as np

import data


class DepthMapNetwork:

    def __init__(self, input_shape, output_shape, learning_rate=0.001):
        self.ckpt_path = os.path.join('.', os.environ['CKPT_DIR'],
                                      '{}'.format(self.__class__.__name__))

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.input = tf.placeholder(tf.float32,
                                        shape=(None, ) + input_shape,
                                        name='input')
            self.target = tf.placeholder(tf.float32,
                                         shape=(None, ) + output_shape,
                                         name='target')
            # Grayscale
            gray = tf.image.rgb_to_grayscale(self.input)

            # Scale to nearest multiple of target size
            resize = tf.image.resize_images(gray,
                                            tuple(itertools.starmap(
                                                lambda x, y: x // y * y,
                                                zip(input_shape, output_shape))
                                            ))

            # Convolve to output size, alternating between horizontal and
            # vertical
            steps_h, steps_v = map(lambda x: x[0] // x[1],
                                   zip(input_shape, output_shape))
            i_boundary = min(steps_h, steps_v) // 2 + 2
            for i in range(i_boundary):
                # Last layer is sigmoid, others relu
                last = i == i_boundary - 1
                conv = tf.layers.conv2d(conv if i > 0 else resize,
                                        1 if last else 32, 3,
                                        strides=(1 + i % 2, 2 - i % 2),
                                        padding='same',
                                        activation=(tf.nn.sigmoid
                                                    if last else
                                                    tf.nn.relu),
                                        name=f'Conv{i}'
                                        )
            self.output = tf.squeeze(conv)

            self.loss = tf.reduce_sum(
                tf.squared_difference(self.output, self.target)
            )
            self.optimizer = tf.train.AdamOptimizer(
                                learning_rate=learning_rate,
                                epsilon=1.0
                             ).minimize(self.loss)

            self.saver = tf.train.Saver()
        self.tb_log = tf.summary.FileWriter(
            os.path.join(
                '.', os.environ['TB_DIR'],
                datetime.now().isoformat()),
            self.graph)

    def __call__(self, dataset, epochs=100, batchsize=32):
        start = time.time()
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())

            for epoch in range(1, 1 + epochs):
                epoch_start = time.time()
                print(f'Epoch: {epoch}')

                for b_in, b_out in data.as_matrix_batches(dataset, batchsize):
                    _, loss, results = s.run(
                        [self.optimizer, self.loss, self.output],
                        {self.input: b_in, self.target: b_out})

                print(f'Elapsed time: {time.time() - start:.3f}',
                      f'Epoch time: {time.time() - epoch_start:.3f}')
                if not epoch % 10:
                    print('Saving')
                    self.saver.save(s, str(self.ckpt_path))

            print('Saving')
            self.saver.save(s, str(self.ckpt_path))

        for i, result in enumerate(results):
            dataset[i].result = result.squeeze()
            if i < 10:
                print(result.min(), result.max())
