import math

import tensorflow as tf

import tfhelper


class _DistributedConvolutionalNeuralFields:
    """Implements Liu et al. (2015): Learning Depth from Single Monocular Images
    Using Deep Convolutional Neural Fields.  DOI: 10.1109/TPAMI.2015.2505283
    """

    def __init__(self):
        self.patch_size = (224, 224)  # Liu et al.: 224x224
        self.sp_size = (10, 10)  # Liu et al.: super pixels, not patches
        self.gamma = 1

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

    @tfhelper.with_scope('extract_superpixel')
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

    @tfhelper.with_scope('extract_patch')
    def patches(self, images):
        # TODO: use superpixels for patch calculation
        patches = tf.extract_image_patches(images=images,
                                           ksizes=[1, *self.patch_size, 1],
                                           strides=[1, *self.sp_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='SAME')
        return tf.reshape(patches, (int(images.shape[0]), -1, *self.patch_size,
                                    int(images.shape[-1])))

    @tfhelper.make_template('unary/patch')
    def unary_part_patch(self, image_patch):
        temp = tf.layers.conv2d(image_patch, 64, 11, activation=tf.nn.relu)
        temp = tf.layers.max_pooling2d(temp, 2, 2)
        temp = tf.layers.conv2d(temp, 256, 5, activation=tf.nn.relu)
        temp = tf.layers.max_pooling2d(temp, 2, 2)
        temp = tf.layers.conv2d(temp, 256, 3, activation=tf.nn.relu)
        temp = tf.layers.conv2d(temp, 256, 3, activation=tf.nn.relu)
        temp = tf.layers.conv2d(temp, 256, 3, activation=tf.nn.relu)
        temp = tf.layers.max_pooling2d(temp, 2, 2)

        # Fit result into dense layer's 1D
        temp = tf.reshape(temp, [int(image_patch.shape[0]), -1])

        temp = tf.layers.dense(temp, 4096, activation=tf.nn.relu)
        temp = tf.layers.dense(temp, 128, activation=tf.nn.relu)
        temp = tf.layers.dense(temp, 16, activation=tf.nn.sigmoid)
        temp = tf.layers.dense(temp, 1, activation=None)
        return temp

    @tfhelper.with_scope('unary')
    def unary_part(self, images):
        patches = self.patches(images)
        return tf.map_fn(self.unary_part_patch, patches)

    @tfhelper.make_template('pairwise/dense')
    def pairwise_dense(self, similarities):
        return tf.layers.dense(similarities, 1, activation=None)

    @tfhelper.with_scope('pairwise/histogram')
    def color_histogram(self, superpixel):
        values = tf.reduce_sum(superpixel * (16777216., 65536., 256.), axis=-1)
        histogram = tf.histogram_fixed_width(values, (0, 16777216.),
                                             256, tf.float32)
        return histogram

    @tfhelper.with_scope('pairwise/similarity')
    def similarity(self, features, pairs):
        left = tf.map_fn(lambda batch: tf.gather(batch, pairs[0]), features)
        right = tf.map_fn(lambda batch: tf.gather(batch, pairs[1]), features)
        return tf.exp(-self.gamma * tf.norm(left - right, axis=2))

    @tfhelper.with_scope('pairwise')
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

    @tfhelper.with_scope('loss')
    def loss_part(self, target, z, r):
        superpixels = self.superpixels(target)
        y_sp = tf.reduce_mean(superpixels, axis=2)
        loss_unary = tf.losses.mean_squared_error(y_sp, z,
                                                  scope='unary')

        pairs = self.pair_indices(target)
        y_p = tf.map_fn(lambda batch: tf.gather(batch, pairs[0]), y_sp)
        y_q = tf.map_fn(lambda batch: tf.gather(batch, pairs[1]), y_sp)

        loss_pairwise = tf.losses.mean_squared_error(y_p, y_q, r,
                                                     scope='pairwise')

        # E(y, x)
        energy = loss_unary + loss_pairwise

        # Neg log-likelihood
        exp_energy = tf.exp(-energy)
        loss = -tf.log(exp_energy / tf.cumsum(exp_energy))
        tf.losses.add_loss(loss)
        return loss

    def __call__(self, images, depths):
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

dcnf = _DistributedConvolutionalNeuralFields()
