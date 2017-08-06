import tensorflow as tf

import tfhelper

class _DistributedConvolutionalNeuralFields:
    """Implements Liu et al. (2015): Learning Depth from Single Monocular Images
    Using Deep Convolutional Neural Fields.  DOI: 10.1109/TPAMI.2015.2505283
    """

    def __init__(self):
        self.patch_size = (224, 224)
        self.sp_size = (10, 10)
        self.gamma = 1

    def patches(self, images):
        # TODO: use superpixels for patch calculation
        patches = tf.extract_image_patches(images=images,
                                           ksizes=[1, *self.patch_size, 1],
                                           strides=[1, *self.sp_size, 1],
                                           rates=[1, 1, 1, 1],
                                           padding='SAME')
        return tf.reshape(patches, (int(images.shape[0]), -1, *self.patch_size,
                                    int(images.shape[-1])))

    @tfhelper.make_template('unary_part_patch')
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

    @tfhelper.with_scope('unary_part')
    def unary_part(self, images):
        patches = self.patches(images)
        return tf.map_fn(self.unary_part_patch, patches)

    def __call__(self, images, depths):
        z = self.unary_part(images)

        output = z
        loss = tf.reduce_mean(output - 1)

        optimizer = tf.train.GradientDescentOptimizer(0.1)
        return optimizer.minimize(loss,
                                  tf.train.get_or_create_global_step())

dcnf = _DistributedConvolutionalNeuralFields()
