import itertools
import os
import signal
import time

from datetime import datetime

import tensorflow as tf
import numpy as np

import data


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


class DepthMapNetwork:

    def setup(__init__):
        def init(self, input_shape, output_shape, *,
                 ckptdir='checkpoints', ckptfreq=50, tbdir='tb_logs',
                 cont=False):
            self.ckpt_path = str(os.path.join('.', ckptdir,
                                              f'{type(self).__name__}'))
            self.ckpt = tf.train.latest_checkpoint(os.path.join('.', ckptdir))
            self.ckptfreq = ckptfreq

            self.cont = cont

            self.graph = tf.Graph()
            with self.graph.as_default():
                self.step = tf.train.create_global_step()

                self.input = tf.placeholder(tf.float32,
                                            shape=(None, ) + input_shape,
                                            name='input')
                self.target = tf.placeholder(tf.float32,
                                             shape=(None, ) + output_shape,
                                             name='target')

                self.optimizer = tf.Print(self.input, [''],
                                          'No self.optimizer implemented', 1)
                self.output = tf.Print(self.input, [''],
                                       'No self.output implemented', 1)
                self.loss = tf.Print(self.input, [''],
                                     'No self.loss implemented', 1)

                __init__(self, input_shape, output_shape)

                (self.epoch_loss,
                 self.epoch_loss_update,
                 self.epoch_loss_reset) = create_reset_metric(
                    tf.contrib.metrics.streaming_mean_squared_error,
                    'epoch_loss',
                    predictions=self.output,
                    labels=self.target)

                with tf.name_scope('loss'):
                    self.summary_loss_train = tf.summary.scalar('train',
                                                                self.loss)
                    self.summary_loss_test = tf.summary.scalar('test',
                                                               self.epoch_loss)

                self.saver = tf.train.Saver()

            self.tb_log = tf.summary.FileWriter(
                os.path.join(
                    '.', tbdir,
                    datetime.now().strftime(
                        f'%m-%dT%H-%M_{type(self).__name__}')),
                self.graph)

        return init

    def test(self, dataset):
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())
            if self.cont:
                self.saver.restore(s, self.ckpt)

            s.run(self.epoch_loss_reset)
            for i, (b_in, b_out) in enumerate(
                    data.as_matrix_batches(dataset, 1, False)):
                feed_dict = {self.input: b_in, self.target: b_out}
                dataset[i].result, loss = s.run([self.output,
                                                 self.epoch_loss_update],
                                                feed_dict)
            s.run(self.epoch_loss_reset)

            print(f'Mean loss is {loss}')

    def train(self, dataset_train, dataset_test, epochs, batchsize):
        start = time.time()
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())
            if self.cont:
                self.saver.restore(s, self.ckpt)
            self.__register_kill_handlers(s)

            for epoch in range(1, 1 + epochs):
                epoch_start = time.time()
                print(f'Starting epoch {epoch}')

                for b_in, b_out in data.as_matrix_batches(dataset_train,
                                                          batchsize):
                    o, loss, step = s.run([self.optimizer,
                                           self.summary_loss_train,
                                           self.step],
                                          {self.input: b_in,
                                           self.target: b_out})
                    self.tb_log.add_summary(loss, step)

                s.run(self.epoch_loss_reset)
                for b_in, b_out in data.as_matrix_batches(
                        dataset_test, batchsize):
                    loss_test, step, loss = s.run([self.summary_loss_test,
                                                   self.step,
                                                   self.epoch_loss_update],
                                                  {self.input: b_in,
                                                   self.target: b_out})
                s.run(self.epoch_loss_reset)

                self.tb_log.add_summary(loss_test, step)
                print(f'Epoch {epoch} finished',
                      f'Elapsed time: {time.time() - start:.3f}',
                      f'Epoch time: {time.time() - epoch_start:.3f}',
                      f'Loss {loss}')
                if not epoch % self.ckptfreq:
                    print(f'Saving checkpoints after epoch {epoch}')
                    self.saver.save(s, self.ckpt_path, global_step=self.step)

            print('Saving final checkpoint')
            self.saver.save(s, self.ckpt_path, global_step=self.step)

    def __register_kill_handlers(self, session):
        def handler(signum, frame):
            print(f'Received signal {signal.Signals(signum).name}, saving...')
            self.saver.save(session, self.ckpt_path, global_step=self.step)
            print(f'Saved successfully.')
        signal.signal(signal.SIGUSR1, handler)
        print('Registered handler for SIGUSR1')
        signal.signal(signal.SIGUSR2, handler)
        print('Registered handler for SIGUSR2')


class DownsampleNetwork(DepthMapNetwork):

    @DepthMapNetwork.setup
    def __init__(self, input_shape, output_shape):
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

        self.loss = tf.losses.mean_squared_error(self.target, self.output)
        self.optimizer = tf.train.AdamOptimizer(
                            learning_rate=0.001,
                            epsilon=1.0
                         ).minimize(self.loss, self.step)
