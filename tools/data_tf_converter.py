import glob
import os
import sys

import numpy as np
import scipy.misc as smisc
import tensorflow as tf


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main():
    datasets = [d for d in sys.argv[1:] if d != '--del_raw']
    if len(datasets) == 0:
        print('Provide a dataset ( data_tf_converter.py DATASET [--del_raw] )')
        exit()
    remove = '--del_raw' in sys.argv

    for dataset in datasets:
        for dir in ['test', 'train']:
            with tf.python_io.TFRecordWriter(
                        os.path.join(os.environ['DATA_DIR'],
                                     dataset,
                                     dir + '.tfrecords')
                    ) as writer:
                directory = os.path.join(os.environ['DATA_DIR'], dataset, dir)
                depth_paths = glob.glob(os.path.join(directory, '*-depth.png'))
                for depth_path in depth_paths:
                    image_path = depth_path[:-9] + 'image.png'
                    depth_img = smisc.imread(depth_path).astype(np.float32) / 255. - .5
                    img = smisc.imread(image_path).astype(np.float32) / 255. - .5
                    name = os.path.splitext(os.path.basename(depth_path))[0][:-6]
                    if len(depth_img.shape) < 3:
                        depth_img = np.expand_dims(depth_img, -1)
                    example = tf.train.Example(
                        features=tf.train.Features(feature={
                            'image_height': _int64_feature(img.shape[0]),
                            'image_width': _int64_feature(img.shape[1]),
                            'image_channels': _int64_feature(img.shape[2]),
                            'depth_height': _int64_feature(depth_img.shape[0]),
                            'depth_width': _int64_feature(depth_img.shape[1]),
                            'depth_channels': _int64_feature(depth_img.shape[2]),
                            'image': _bytes_feature(img.tobytes()),
                            'depth': _bytes_feature(depth_img.tobytes()),
                        })
                    )
                    writer.write(example.SerializeToString())
                    if remove:
                        os.remove(depth_path)
                        os.remove(image_path)


if __name__ == '__main__':
    main()
