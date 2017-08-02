import tensorflow as tf

def DistributedConvolutionalNeuralFields():
    input_layer = tf.random_normal([2, 5])
    output = tf.layers.dense(input_layer, 3, activation=tf.nn.relu)
    return tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_sum(output), tf.train.get_or_create_global_step())

dcnf = DistributedConvolutionalNeuralFields
