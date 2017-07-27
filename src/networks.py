import functools
import itertools
import logging
import math
import os
import signal
import sys
import time

from datetime import datetime

import tensorflow as tf
import numpy as np

import data

logger = logging.getLogger('ann3depth')


def create_reset_metric(metric, scope='reset_metrics', **metric_args):
    with tf.variable_scope(scope) as scope:
        metric_op, update_op = metric(**metric_args)
        vars = tf.contrib.framework.get_variables(
            scope, collection=tf.GraphKeys.LOCAL_VARIABLES)
        reset_op = tf.variables_initializer(vars)
    return metric_op, update_op, reset_op


def make_template(scope=None, create_scope_now_=False, unique_name_=None,
                  custom_getter_=None, **kwargs):
    def make_tf_template(function):
        template = tf.make_template(function.__name__
                                    if scope is None or callable(scope)
                                    else scope,
                                    function,
                                    create_scope_now_=create_scope_now_,
                                    unique_name_=unique_name_,
                                    custom_getter_=custom_getter_,
                                    **kwargs)

        @functools.wraps(function)
        def wrapper(*caller_args, **caller_kwargs):
            return template(*caller_args, **caller_kwargs)
        return wrapper

    if callable(scope):
        return make_tf_template(scope)
    return make_tf_template


def with_scope(scope):
    def add_scope(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            with tf.name_scope(scope):
                return function(*args, **kwargs)
        return wrapper
    return add_scope


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

                with tf.name_scope('Warnings'):
                    self.optimizer = tf.Print(
                        self.input, [''], 'No self.optimizer implemented', 1)
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

                with tf.name_scope('train_img'):
                    self.summary_input_train = tf.summary.image(
                        'input', self.input)
                    self.summary_output_train = tf.summary.image(
                        'output', self.output)
                    self.summary_target_train = tf.summary.image(
                        'target', tf.expand_dims(self.target, -1))

                with tf.name_scope('test_img'):
                    self.summary_input_test = tf.summary.image(
                        'input', self.input)
                    self.summary_output_test = tf.summary.image(
                        'output', self.output)
                    self.summary_target_test = tf.summary.image(
                        'target', tf.expand_dims(self.target, -1))

                self.summary_train = tf.summary.merge(
                    [self.summary_input_train,
                     self.summary_output_train,
                     self.summary_target_train])
                self.summary_test = tf.summary.merge([self.summary_input_test,
                                                      self.summary_output_test,
                                                      self.summary_target_test])
                self.saver = tf.train.Saver()

            if self.cont:  # Select old filewriter path
                try:
                    directories = sorted(os.listdir(
                            os.path.join('.', tbdir, type(self).__name__)))
                    directories = [s for s in directories if s[0].isdigit()]
                    filewriter_path = os.path.join(
                        '.', tbdir, type(self).__name__, directories[-1])
                except (FileNotFoundError, IndexError):
                    filewriter_path = None

            if not self.cont or not filewriter_path:
                filewriter_path = os.path.join('.', tbdir,
                                               type(self).__name__,
                                               datetime.now().strftime(
                                                   f'%m-%dT%H-%M'))

            logger.info(f'FileWriter writes to {filewriter_path}.')
            self.tb_log = tf.summary.FileWriter(filewriter_path,
                                                self.graph
                                                if not self.cont else
                                                None)
            exit()

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
                dataset[i].result, loss = s.run([tf.squeeze(self.output),
                                                 self.epoch_loss_update],
                                                feed_dict)
            s.run(self.epoch_loss_reset)

            logger.info(f'Mean loss is {loss}')

    def train(self, dataset_train, dataset_test, epochs, batchsize):
        start = time.time()
        with tf.Session(graph=self.graph) as s:
            s.run(tf.global_variables_initializer())
            if self.cont:
                self.saver.restore(s, self.ckpt)
            step = s.run(self.step)
            start_epoch = 1 + math.ceil(step / (len(dataset_train) / batchsize))
            self.__register_kill_handlers(s)

            self.tb_log.add_session_log(
                tf.SessionLog(status=tf.SessionLog.START),
                global_step=step)

            logger.debug(f'First epoch is {start_epoch}')
            for epoch in range(start_epoch, 1 + epochs):
                epoch_start = time.time()
                logger.info(f'Starting epoch {epoch}')

                first_batch = True
                for b_in, b_out in data.as_matrix_batches(dataset_train,
                                                          batchsize):

                    run_metadata = tf.RunMetadata()
                    run_options = tf.RunOptions(
                        trace_level=tf.RunOptions.FULL_TRACE
                                    if first_batch else
                                    tf.RunOptions.NO_TRACE)
                    o, loss_train, summary_train, step = s.run(
                        [self.optimizer,
                         self.summary_loss_train,
                         self.summary_train,
                         self.step],
                        {self.input: b_in,
                         self.target: b_out},
                        run_options,
                        run_metadata)
                    self.tb_log.add_summary(loss_train, step)
                    if first_batch:
                        try:
                            self.tb_log.add_run_metadata(
                                run_metadata, f'Epoch {epoch}', step)
                        except ValueError as ve:
                            logger.warning('Metadata not stored: %s', ve)
                        first_batch = False

                s.run(self.epoch_loss_reset)
                for b_in, b_out in data.as_matrix_batches(
                        dataset_test, batchsize):
                    loss_test, step, loss, summary_test = s.run(
                        [self.summary_loss_test,
                         self.step,
                         self.epoch_loss_update,
                         self.summary_test],
                        {self.input: b_in,
                         self.target: b_out})
                s.run(self.epoch_loss_reset)

                self.tb_log.add_summary(loss_test, step)

                if epoch % 100 == 1:
                    self.tb_log.add_summary(summary_train, step)
                    self.tb_log.add_summary(summary_test, step)

                logger.info(f'Epoch {epoch} finished; ' +
                            f'Elapsed time: {time.time() - start:.3f}; ' +
                            f'Epoch time: {time.time() - epoch_start:.3f}; ' +
                            f'Loss {loss}')

                if not epoch % self.ckptfreq:
                    logger.info(f'Saving checkpoints after epoch {epoch}')
                    self.tb_log.add_session_log(
                        tf.SessionLog(status=tf.SessionLog.CHECKPOINT),
                        global_step=step)
                    self.saver.save(s, self.ckpt_path, global_step=self.step)

            logger.info('Saving final checkpoint')
            self.saver.save(s, self.ckpt_path, global_step=self.step)
            self.tb_log.add_session_log(
                tf.SessionLog(status=tf.SessionLog.STOP),
                global_step=step)

    def __register_kill_handlers(self, session):
        def handler(signum, frame):
            logger.critical(f'Received signal {signal.Signals(signum).name}')
            self.saver.save(session, self.ckpt_path, global_step=self.step)
            step = session.run(self.step)
            self.tb_log.add_session_log(
                tf.SessionLog(status=tf.SessionLog.CHECKPOINT),
                global_step=step)
            logger.info(f'Saved successfully. Shutting down...')
            sys.exit('Shut down after receiving signal ' +
                     f'{signal.Signals(signum).name}')

        for s in [signal.SIGUSR1, signal.SIGUSR2, signal.SIGALRM,
                  signal.SIGINT]:
            signal.signal(s, handler)
            logger.debug(f'Registered handler for {s.name}')


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
        self.output = conv

        self.loss = tf.losses.mean_squared_error(self.target,
                                                 tf.squeeze(self.output))
        self.optimizer = tf.train.AdamOptimizer(
                            learning_rate=0.001,
                            epsilon=1.0
                         ).minimize(self.loss, self.step)


class DeepConvolutionalNeuralFields(DepthMapNetwork):

    def _create_layers(self, name, layer_in, layers):
        for i, layer in enumerate(layers):
            if layer['type'] == 'conv2d':
                layer_in = tf.layers.conv2d(
                    inputs=layer_in, filters=layer['filters'],
                    kernel_size=layer['size'],
                    padding='same', activation=layer['act'],
                    name=f"{name.title()}Conv_{layer['size']}x{layer['size']}_{i}")
            elif layer['type'] == 'pool':
                layer_in = tf.layers.max_pooling2d(
                    inputs=layer_in,
                    pool_size=layer['pool_size'],
                    strides=layer['strides'],
                    name=f'{name.title()}Pool_{i}')
            elif layer['type'] == 'fc':
                layer_in = tf.layers.dense(inputs=layer_in,
                                           units=layer['size'],
                                           activation=layer['act'],
                                           name=f'{name.title()}Dense_{i}')
            elif layer['type'] == 'reshape':
                layer_in = tf.reshape(tensor=layer_in,
                                      shape=layer['shape'],
                                      name=f'{name.title()}Reshape_{i}')
        return layer_in

    @make_template
    def pairwise_part(self, layer_in):
        layers = [
            {'type': 'fc', 'size': 1, 'act': None}
        ]
        return self._create_layers('Pairwise', layer_in, layers)

    @make_template
    def unary_part(self, layer_in):
        layers = [
            {'type': 'conv2d', 'size': 11, 'act': tf.nn.relu, 'filters': 64},
            {'type': 'pool', 'pool_size': (2, 2), 'strides': (2, 2)},
            {'type': 'conv2d', 'size': 5, 'act': tf.nn.relu, 'filters': 256},
            {'type': 'pool', 'pool_size': (2, 2), 'strides': (2, 2)},
            {'type': 'conv2d', 'size': 3, 'act': tf.nn.relu, 'filters': 256},
            {'type': 'conv2d', 'size': 3, 'act': tf.nn.relu, 'filters': 256},
            {'type': 'conv2d', 'size': 3, 'act': tf.nn.relu, 'filters': 256},
            {'type': 'pool', 'pool_size': (2, 2), 'strides': (2, 2)},
            {'type': 'reshape', 'shape': [-1, 1]},
            {'type': 'fc', 'size': 4096, 'act': tf.nn.relu},
            {'type': 'fc', 'size': 128, 'act': tf.nn.relu},
            {'type': 'fc', 'size': 16, 'act': tf.nn.sigmoid},
            {'type': 'fc', 'size': 1, 'act': None},
        ]
        return self._create_layers('Unary', layer_in, layers)

    def _create_unary_part(self, superpixels):
        with tf.name_scope('UnaryPart'):
            collection = []
            for sp in superpixels:
                collection.append(self.unary_part(sp))
            return tf.concat(collection, 0, 'UnaryConcat')

    def _create_pairwise_part(self, superpixels):
        with tf.name_scope('PairwisePart'):
            collection = []
            for i, sp_l in enumerate(superpixels):
                for sp_r in superpixels[i+1:]:
                    similarity = self._similarity(sp_l, sp_r)
                    collection.append(self.pairwise_part(similarity))
            return tf.concat(collection, 0, 'PairwiseConcat')

    def _similarity_color_diff(self, superpixel_p, superpixel_q):
        with tf.name_scope('color_diff'):
            diff = superpixel_p - superpixel_q
            return tf.reduce_mean(tf.reduce_mean(diff, 1), 1)

    def _similarity_color_hist(self, superpixel_p, superpixel_q):
        with tf.name_scope('color_hist'):
            # These floats are equivalent to 0x000000, and 0xffffff
            value_range = (0., 16777215.)
            bins = 100

            def flat_color(superpixel):
                # These floats are equivalent to 0xff0000, 0x00ff00, 0x0000ff
                return tf.tensordot(superpixel, (16711680., 65280., 255.), 3)

            def make_histogram(superpixel):
                values = flat_color(superpixel)
                return tf.histogram_fixed_width(values=values,
                                                value_range=value_range,
                                                nbins=bins)

            hist_p = tf.map_fn(make_histogram, superpixel_p)
            hist_q = tf.map_fn(make_histogram, superpixel_q)

            return tf.to_float(hist_p - hist_q)

    def _similarity_local_bin_pattern(self, superpixel_p, superpixel_q):
        # TODO: DOI: 10.1109/ICPR.1994.576366
        with tf.name_scope('local_bin_pattern'):
            return tf.reduce_sum(
                tf.reduce_sum(
                    tf.zeros_like(superpixel_p), 1), 1)

    def _similarity(self, superpixel_p, superpixel_q, gamma=1):
        with tf.name_scope('Similarity'):
            sim = lambda x: tf.exp(-gamma * tf.norm(x, 1))
            similarities = [
                tf.map_fn(sim, self._similarity_color_diff(superpixel_p,
                                                           superpixel_q)),
                tf.map_fn(sim, self._similarity_color_hist(superpixel_p,
                                                           superpixel_q)),
                tf.map_fn(sim, self._similarity_local_bin_pattern(superpixel_p,
                                                                  superpixel_q))
            ]
            concat = tf.concat(similarities, 0, 'SimilarityConcat')
            return tf.expand_dims(concat, 1)

    @DepthMapNetwork.setup
    def __init__(self, input_shape, output_shape):
        # TODO: over segmentation instead of dummies
        superpixels = [tf.placeholder(tf.float32,
                                      shape=(None, 224, 224, 3),
                                      name='superpixel') for x in range(5)]
        z = self._create_unary_part(superpixels)
        r = self._create_pairwise_part(superpixels)

        # TODO: loss function, optimizer
        self.output = self.target
