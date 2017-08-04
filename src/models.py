import tensorflow as tf

def DistributedConvolutionalNeuralFields(images, depths):
    output = tf.layers.conv2d(images, 1, 3)
    output = tf.image.resize_images(output, depths.shape[1:3])
    tf.summary.image('test_in', images, max_outputs=1)
    tf.summary.image('test_out', output, max_outputs=1)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    return optimizer.minimize(tf.reduce_sum(output - depths),
                              tf.train.get_or_create_global_step())

dcnf = DistributedConvolutionalNeuralFields
