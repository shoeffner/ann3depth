import math

import tensorflow as tf

import tfhelper


class _DistributedConvolutionalNeuralFields:
    """Implements Liu et al. (2015): Learning Depth from Single Monocular Images
    Using Deep Convolutional Neural Fields.  DOI: 10.1109/TPAMI.2015.2505283
    """

    def __init__(self):
        self.patch_size = (100, 100)  # Liu et al.: 224x224
        self.sp_size = (40, 40)  # Liu et al.: super pixels, not patches
        self.gamma = 1
        self.epsilon = 1e-7  # For numerical stability

    def pair_indices(self, images):
        max_rows, max_cols = self.num_superpixels(images)
        left = []
        right = []
        for row in range(1, max_rows - 1):
            for col in range(2 - (row & 1), max_cols - 1, 2):
                pixel = row * max_cols + col
                for addend in [-max_cols, max_cols, -1, 1]:
                    left.append(pixel)
                    right.append(pixel + addend)
        return left, right

    def num_superpixels(self, images):
        rows = math.ceil(int(images.shape[1]) / self.sp_size[0])
        cols = math.ceil(int(images.shape[2]) / self.sp_size[1])
        return rows, cols

    @tfhelper.variable_scope('extract_superpixel')
    def superpixels(self, images):
        # TODO: over-segmentation instead of simple extraction
        superpixels = tf.extract_image_patches(images=images,
                                               ksizes=[1, *self.sp_size, 1],
                                               strides=[1, *self.sp_size, 1],
                                               rates=[1, 1, 1, 1],
                                               padding='SAME')
        return tf.reshape(superpixels, (int(images.shape[0]),
                                        -1,
                                        self.sp_size[0] * self.sp_size[1],
                                        int(images.shape[-1])))

    @tfhelper.variable_scope('extract_patch')
    def patches(self, images):
        # TODO: use superpixels for patch calculation
        patches = tf.extract_image_patches(images=images,
                                           ksizes=[1, *self.patch_size, 1],
                                           strides=[1, *self.sp_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='SAME')
        return tf.reshape(patches, (int(images.shape[0]), -1, *self.patch_size,
                                    int(images.shape[-1])))

    @tfhelper.make_template('unary_layers')
    def unary_part_patch(self, image_patch):
        with tf.device('/job:worker/task:1'):
            temp = tf.layers.conv2d(image_patch, 64, 11, activation=tf.nn.relu)
            temp = tf.layers.max_pooling2d(temp, 2, 2)
        with tf.device('/job:worker/task:2'):
            temp = tf.layers.conv2d(temp, 256, 5, activation=tf.nn.relu)
            temp = tf.layers.max_pooling2d(temp, 2, 2)
            temp = tf.layers.conv2d(temp, 256, 3, activation=tf.nn.relu)
        with tf.device('/job:worker/task:3'):
            temp = tf.layers.conv2d(temp, 256, 3, activation=tf.nn.relu)
            temp = tf.layers.conv2d(temp, 256, 3, activation=tf.nn.relu)
            temp = tf.layers.max_pooling2d(temp, 2, 2)

        # Fit result into dense layer's 1D
        temp = tf.reshape(temp, [int(image_patch.shape[0]), -1])

        # temp = tf.layers.dense(temp, 4096, activation=tf.nn.relu)
        with tf.device('/job:worker/task:2'):
            temp = tf.layers.dense(temp, 128, activation=tf.nn.relu)
            temp = tf.layers.dense(temp, 16, activation=tf.nn.sigmoid)
            temp = tf.layers.dense(temp, 1, activation=None)
        return temp

    @tfhelper.variable_scope('unary',
                         partitioner=tf.variable_axis_size_partitioner((64 << 20) -1))
    def unary_part(self, images):
        patches = self.patches(images)
        return tf.map_fn(self.unary_part_patch, patches)

    @tfhelper.make_template('pairwise_layers')
    def pairwise_dense(self, similarities):
        return tf.layers.dense(similarities, 1, activation=None)

    @tfhelper.variable_scope('histogram')
    def color_histogram(self, superpixel):
        values = tf.reduce_sum(superpixel * (16777216., 65536., 256.), axis=-1)
        histogram = tf.histogram_fixed_width(values, (0, 16777216.),
                                             256, tf.float32)
        return histogram

    @tfhelper.variable_scope('similarity')
    def similarity(self, features, pairs):
        left = tf.map_fn(lambda batch: tf.gather(batch, pairs[0]), features)
        right = tf.map_fn(lambda batch: tf.gather(batch, pairs[1]), features)
        return tf.exp(-self.gamma * tf.norm(left - right, axis=2))

    @tfhelper.variable_scope('pairwise',
                         partitioner=tf.variable_axis_size_partitioner((64 << 20) -1))
    def pairwise_part(self, images):
        superpixels = self.superpixels(images)
        pairs = self.pair_indices(images)

        histograms = tf.map_fn(lambda batch: tf.map_fn(self.color_histogram,
                                                    batch), superpixels)

        # color difference similarity
        cdiff_sim = self.similarity(tf.reduce_mean(superpixels, axis=-1),
                                    pairs)
        # color histogram similarity
        histdiff_sim = self.similarity(histograms, pairs)
        # TODO: texture disparity

        # gather similarities
        similarities = tf.stack([cdiff_sim, histdiff_sim], axis=-1)

        return tf.map_fn(self.pairwise_dense, similarities)

    @tfhelper.variable_scope('loss')
    def loss_part(self, target, z, r):
        superpixels = self.superpixels(target)
        y = tf.reduce_mean(superpixels, axis=2)
        pairs = self.pair_indices(target)

        # See Liu et al. (2015) p. 5, eq. (9) - (11)
        def get_A(batch_item, R):
            I = tf.eye(int(R.shape[0]))
            R = tf.scatter_nd_update(R, list(zip(*pairs)),
                                     tf.squeeze(batch_item))
            R = tf.scatter_nd_update(R, list(zip(*pairs[::-1])),
                                     tf.squeeze(batch_item))
            D = tf.diag(tf.reduce_sum(R, axis=1))
            return I + D - R

        # Get reused helpers for loss calculation
        with tf.name_scope('calc_A'):
            # Define R outside of get_A to avoid UPDATE_OP being placed inside
            # loop (see https://github.com/tensorflow/tensorflow/issues/6087)
            R = tf.Variable(tf.zeros((int(r.shape[1]), int(r.shape[1]))),
                            trainable=False)
            A = tf.map_fn(lambda y: get_A(y, R), r,
                          parallel_iterations=1)

        zT = tf.transpose(z, [0, 2, 1])
        yT = tf.transpose(y, [0, 2, 1])

        # energy = E(y, x)
        with tf.name_scope('energy'):
            energy = tf.squeeze(yT @ A @ y - 2 * zT @ y + zT @ z)

        # Integral Z(x) = exp( -E(y, x) ) dy
        with tf.name_scope('integral'):
            fac = math.pi ** (int(r.shape[1]) / 2)
            fac /= (tf.matrix_determinant(A) ** .5) + self.epsilon
            inverseA = tf.matrix_inverse(A) + self.epsilon
            exp = tf.squeeze(tf.exp(zT @ inverseA @ z - zT @ z))
            Z = fac * exp + self.epsilon

        # Neg log-likelihood
        with tf.name_scope('nll'):
            loss = -tf.log(tf.exp(-energy) / Z + self.epsilon)

        # Mean over batch
        loss = tf.reduce_mean(loss, name='mean_loss')
        tf.losses.add_loss(loss)

        return loss

    def __call__(self, images, depths):
        images = tf.image.resize_images(images, [240, 320])
        depths = tf.image.resize_images(depths, [240, 320])

        z = self.unary_part(images)
        r = self.pairwise_part(images)
        loss = self.loss_part(depths, z, r)

        with tf.name_scope('output'):
            rows, cols = self.num_superpixels(images)
            output = tf.image.resize_images(tf.reshape(z, [-1, rows, cols, 1]),
                                            (int(depths.shape[1]),
                                            int(depths.shape[2])))

        with tf.name_scope('summaries'):
            tf.summary.image('Output', output, max_outputs=1)
            tf.summary.image('Input', images, max_outputs=1)
            tf.summary.image('Target', depths, max_outputs=1)

        optimizer = tf.train.GradientDescentOptimizer(0.1)
        return optimizer.minimize(loss,
                                  tf.train.get_or_create_global_step())


class _MNISTTest:
    def __call__(self, inputs, targets):
        temp = tf.reshape(inputs, [-1, 784])
        temp = tf.layers.dense(temp, 8, activation=tf.nn.relu)
        temp = tf.layers.dense(temp, 4, activation=tf.nn.relu)
        temp = tf.layers.dense(temp, 2, activation=tf.nn.relu)
        temp = tf.layers.dense(temp, 1, activation=tf.nn.sigmoid)

        with tf.name_scope('summaries'):
            output = tf.cast(temp, tf.uint8)
            tf.summary.scalar('Output', output)
            tf.summary.image('Input', inputs, max_outputs=1)
            tf.summary.scalar('Target', targets)

        optimizer = tf.train.AdamOptimizer()
        loss = tf.losses.mean_squared_error(tf.cast(targets, tf.float32), temp)
        return optimizer.minimize(loss,
                                  tf.train.get_or_create_global_step())


class _MultiScaleDeepNetwork:
    """Implements Eigen et al. (2014): Depth Map Prediction from a Single Image
    using a Multi-Scale Deep Network.
    """

    @tfhelper.variable_scope('coarse')
    def coarse(self, images):
        with tf.name_scope('conv'):
            temp = tf.layers.conv2d(images, 96, 11, (4, 4), activation=tf.nn.relu)
            temp = tf.layers.max_pooling2d(temp, 2, 2)
            temp = tf.layers.conv2d(temp, 256, 5, padding='same', activation=tf.nn.relu)
            temp = tf.layers.max_pooling2d(temp, 2, 2)
            temp = tf.layers.conv2d(temp, 384, 3, padding='same', activation=tf.nn.relu)
            temp = tf.layers.conv2d(temp, 384, 3, padding='same', activation=tf.nn.relu)
            # Note: Added striding to achieve same activation size
            temp = tf.layers.conv2d(temp, 256, 3, (2, 2), activation=tf.nn.relu)

            temp = tf.reshape(temp, [int(temp.shape[0]), -1])

        with tf.name_scope('dense'):
            temp = tf.layers.dense(temp, 4096, activation=tf.nn.relu)
            temp = tf.layers.dropout(temp, training=True)
            temp = tf.layers.dense(temp, 55 * 74, activation=None)

            temp = tf.reshape(temp, [-1, 55, 74, 1])

        return temp

    @tfhelper.variable_scope('fine')
    def fine(self, images, coarse):
        temp = tf.layers.conv2d(images, 63, 9, (2, 2), activation=tf.nn.relu, name='first')
        temp = tf.layers.max_pooling2d(temp, 2, 2)

        temp = tf.concat([temp, coarse], axis=-1)

        temp = tf.layers.conv2d(temp, 64, 5, padding='same', activation=tf.nn.relu, name='second')

        temp = tf.layers.conv2d(temp, 1, 5, padding='same', activation=None, name='third')

        return temp

    def loss(self, outputs, targets, name):
        outputs = tf.reshape(outputs, [int(outputs.shape[0]), -1])
        targets = tf.reshape(targets, [int(targets.shape[0]), -1])

        eps = 1e-8
        lambd = 0.5
        log_out = tf.log(outputs + eps)
        log_out = tf.where(tf.is_nan(log_out), tf.zeros_like(log_out), log_out)
        log_tar = tf.log(targets + eps)
        log_tar = tf.where(tf.is_nan(log_tar), tf.zeros_like(log_tar), log_tar)

        l2norm = tf.losses.mean_squared_error(log_out, log_tar,
                                              loss_collection=[])
        scaleinv = tf.square(tf.reduce_sum(log_out - log_tar, 1))

        loss = l2norm - lambd / (74 * 55)  * scaleinv

        # Mean loss for batch
        loss = tf.reduce_mean(loss, name=name)

        tf.losses.add_loss(loss)
        return loss


    def __call__(self, images, depths):
        with tf.name_scope('preprocessing'):
            images = tf.image.resize_images(images, [228, 304])
            depths = tf.image.resize_images(depths, [55, 74])

        coarse = self.coarse(images)
        outputs = self.fine(images, coarse)

        with tf.name_scope('loss'):
            loss_coarse = self.loss(coarse, depths, 'coarse_loss')
            loss_fine = self.loss(outputs, depths, 'fine_loss')


        with tf.name_scope('optimizers'):
            samples_coarse = 2000000
            samples_fine = 1500000
            batch_size = int(images.shape[0])

            steps_coarse = samples_coarse // batch_size
            steps_fine = samples_fine // batch_size

            global_step = tf.train.get_or_create_global_step()
            train_vars = tf.GraphKeys.TRAINABLE_VARIABLES

            def create_optimizer(loss, rate, momentum, collections,
                                 name='Momentum', global_step=None):
                optimizer = tf.train.MomentumOptimizer(rate, momentum,
                                                       name=name)
                vars = []
                for c in collections:
                    vars += tf.get_collection(train_vars, c)
                return optimizer.minimize(loss, global_step=global_step,
                                          var_list=vars)

            def coarse_optimizers():
                return [
                    create_optimizer(loss_coarse, 0.001, 0.9,
                                     ['coarse/conv'], 'CoarseConv',
                                     global_step),
                    create_optimizer(loss_coarse, 0.1, 0.9,
                                     ['coarse/dense'], 'CoarseDense', None)
                ]

            def fine_optimizers():
                return [
                    create_optimizer(loss_fine, 0.001, 0.9,
                                     ['fine/first', 'fine/third'], 'FineA',
                                     global_step),
                    create_optimizer(loss_fine, 0.01, 0.9,
                                     ['fine/second'], 'FineB', None)
                ]

            cond_fine = global_step <= (steps_coarse + steps_fine)
            cond_coarse = global_step <= steps_coarse

            fine_train = tf.cond(cond_fine,
                                 fine_optimizers,
                                 # TODO: keep updating global step!
                                 lambda: [tf.no_op(), tf.no_op()],
                                 name='Fine')
            coarse_train = tf.cond(cond_coarse,
                                   coarse_optimizers,
                                   lambda: fine_train,
                                   name='Coarse')

        with tf.name_scope('summaries'):
            tf.summary.image('Input', images, max_outputs=3)
            tf.summary.image('Coarse', coarse, max_outputs=3)
            tf.summary.image('Fine', outputs, max_outputs=3)
            tf.summary.image('Target', depths, max_outputs=3)
            phase = tf.cond(cond_fine, lambda: 2, lambda: 3)
            phase = tf.cond(cond_coarse, lambda: 1, lambda: phase)
            tf.summary.scalar('Phase', phase)

        return coarse_train


dcnf = _DistributedConvolutionalNeuralFields()
mnist = _MNISTTest()
msdn = _MultiScaleDeepNetwork()
