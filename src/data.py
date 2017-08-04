import glob
import os

import tensorflow as tf

import tfhelper


def _read(queue):
    reader = tf.TFRecordReader()
    _, record = reader.read(queue)
    example = tf.parse_single_example(record, features={
                    'image_height': tf.FixedLenFeature([], tf.int64),
                    'image_width': tf.FixedLenFeature([], tf.int64),
                    'image_channels': tf.FixedLenFeature([], tf.int64),
                    'depth_height': tf.FixedLenFeature([], tf.int64),
                    'depth_width': tf.FixedLenFeature([], tf.int64),
                    'depth_channels': tf.FixedLenFeature([], tf.int64),
                    'image': tf.FixedLenFeature([], tf.string),
                    'depth': tf.FixedLenFeature([], tf.string),
                })
    image = tf.decode_raw(example['image'], tf.float32)
    depth = tf.decode_raw(example['depth'], tf.float32)
    image = tf.reshape(image, [480, 640, 3])
    depth = tf.reshape(depth, [480, 640, 1])
    return image, depth


@tfhelper.with_scope('input')
def inputs(datasets, batch_size=32):
    paths = []
    for dataset in datasets:
        paths.append(
            os.path.join(os.environ['DATA_DIR'], dataset, 'train.tfrecords')
        )
    queue = tf.train.string_input_producer(paths)

    image, depth = _read(queue)

    return tf.train.shuffle_batch([image, depth],
                                  batch_size=batch_size,
                                  capacity=4*batch_size,
                                  min_after_dequeue=2*batch_size,
                                  num_threads=2)
