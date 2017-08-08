import collections
import os

import tensorflow as tf

import tfhelper


Pipeline = collections.namedtuple('Pipeline', ('files',
                                               'labels',
                                               'reader',
                                               'convert'))


def _get_pipeline(dataset):
    default = Pipeline(files=_files_tfrecords,
                       labels=None,
                       reader=lambda q: _read(q, tf.TFRecordReader()),
                       convert=_convert_img_depth)
    mnist = Pipeline(files=lambda *a: _files_list_dir(*a, filter='images'),
                     labels=lambda *a: _files_list_dir(*a, filter='labels'),
                     reader=lambda q0, q1:
                         _read(q0, tf.FixedLengthRecordReader(28 * 28, 16),
                               q1, tf.FixedLengthRecordReader(1, 8)),
                     convert=_convert_mnist)
    pipelines = {
        'make3d1': default,
        'make3d2': default,
        'mnist': mnist,
        'nyu': default,
    }
    return pipelines.get(dataset, default)


@tfhelper.with_scope('input')
def inputs(datadir, dataset, batch_size=32, train_or_test='train'):
    pipeline = _get_pipeline(dataset)
    base_dir = os.path.join(datadir, dataset)

    files = pipeline.files(base_dir, train_or_test)
    queue_files = tf.train.string_input_producer(files)

    if not pipeline.labels:
        record = pipeline.reader(queue_files)
    else:
        labels = pipeline.labels(base_dir)
        queue_labels = tf.train.string_input_producer(labels)

        record = pipeline.reader(queue_files, queue_labels)

    input_, target = pipeline.convert(*record)
    return tf.train.shuffle_batch([input_, target],
                                  batch_size=batch_size,
                                  capacity=4*batch_size,
                                  min_after_dequeue=2*batch_size,
                                  num_threads=2)


def _files_tfrecords(base_dir, train_or_test='train'):
    return [os.path.join(base_dir, f'{train_or_test}.tfrecords')]


def _files_list_dir(base_dir, train_or_test='train', filter=''):
    dir = os.path.join(base_dir, train_or_test)
    return [os.path.join(dir, p) for p in os.listdir(dir) if filter in p]


def _read(queue, queue_reader, label_queue=None, label_reader=None):
    key, record = queue_reader.read(queue)
    label = None
    if label_queue and label_reader:
        _, label = label_reader.read(label_queue)
    return key, record, label


def _convert_img_depth(key, record, *args):
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

    # Make static size
    image = tf.image.resize_images(image, [240, 320])
    depth = tf.image.resize_images(depth, [240, 320])
    return image, depth


def _convert_mnist(key, record, label):
    image = tf.parse_single_example(record, features={
                'data': tf.FixedLenFeature([], tf.string),
            })
    image = tf.decode_raw(image['data'], tf.uint8)
    image = tf.reshape(image, [28, 28, 1])

    # Make static size
    image = tf.image.resize_images(image, [28, 28])
    image = tf.cast(image, tf.float32) / 255. - .5
    label = tf.parse_single_example(record, features={
                'label': tf.FixedLenFeature([], tf.string),
            })
    label = tf.decode_raw(label['label'], tf.uint8)
    label = tf.reshape(label, [1])
    return image, label
