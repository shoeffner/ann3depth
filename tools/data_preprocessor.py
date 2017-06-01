import os
import sys

import h5py

import numpy as np
import scipy.io as sio
import scipy.misc as smisc

WIDTH = 640
HEIGHT = 480
D_WIDTH = 55 * WIDTH // HEIGHT
D_HEIGHT = 55

START = 125
LIMIT = None


def include(img):
    return img[img.index('.') + 1:] not in ['txt']


def __process_make3d1(path_train, path_test):
    path = os.path.join('data', 'make3d1', 'unpacked')
    depth_path = [os.path.join(path, d) for d in ['Train400Depth',
                                                  'Test134Depth']]
    img_path = [os.path.join(path, d) for d in ['Train400Img',
                                                'Test134']]

    target_path = [path_train, path_test]

    for dp, ip, tp in zip(depth_path, img_path, target_path):
        print(f'Preprocessing images in {dp} and {ip}')

        depths = list(filter(include, os.listdir(dp)))
        imgs = list(filter(include, os.listdir(ip)))

        c = START
        for d, i in zip(depths[START:LIMIT], imgs[START:LIMIT]):
            try:
                name = d[d.index('-') + 1:d.index('.')]

                img = smisc.imread(os.path.join(ip, i))
                img = smisc.imresize(img, (WIDTH, HEIGHT))

                depth = sio.loadmat(os.path.join(dp, d))
                depth = depth['Position3DGrid'][..., 3]

                depth = smisc.imresize(depth, (D_WIDTH, D_HEIGHT))
            except ValueError as ve:
                print(f'Skipping sample {c}, {d} and {i}. Reason: {ve}')
                continue
            c += 1

            smisc.imsave(os.path.join(tp, f'{name}-image.png'), img)
            smisc.imsave(os.path.join(tp, f'{name}-depth.png'), depth)


def __process_make3d2(path_train, path_test):
    path = os.path.join('data', 'make3d2', 'unpacked')

    depth_path = [os.path.join(path, d) for d in ['Dataset3_Depths',
                                                  'Dataset2_Depths']]
    img_path = [os.path.join(path, d) for d in ['Dataset3_Images',
                                                'Dataset2_Images']]
    target_path = [path_train, path_test]

    for dp, ip, tp in zip(depth_path, img_path, target_path):
        print(f'Preprocessing images in {dp} and {ip}')

        depths = list(filter(include, os.listdir(dp)))
        imgs = list(filter(include, os.listdir(ip)))

        c = START
        for d, i in zip(depths[START:LIMIT], imgs[START:LIMIT]):
            try:
                name = d[d.index('-') + 1:d.index('.')]

                img = smisc.imread(os.path.join(ip, i))
                img = np.rot90(img, k=-1)
                img = smisc.imresize(img, (WIDTH, HEIGHT))

                depth = sio.loadmat(os.path.join(dp, d))['depthMap']
                depth = smisc.imresize(depth, (D_WIDTH, D_HEIGHT))
            except ValueError as ve:
                print(f'Skipping sample {c}, {d} and {i}. Reason: {ve}')
                continue
            c += 1

            smisc.imsave(os.path.join(tp, f'{name}-image.png'), img)
            smisc.imsave(os.path.join(tp, f'{name}-depth.png'), depth)


def __process_nyu(path_train, path_test):
    target_path = [path_train, path_test]

    train_images = 5

    path = os.path.join('data', 'nyu', 'unpacked', 'nyu_depth_v2_labeled.mat')

    c = 0
    with h5py.File(path) as mat:
        for d, i, n in zip(mat['depths'],
                           mat['images'],
                           mat['rawRgbFilenames'][0]):
            if c < START:
                c += 1
                continue

            img = smisc.imresize(i, (WIDTH, HEIGHT))
            depth = smisc.imresize(d, (D_WIDTH, D_HEIGHT))

            name = (''.join(map(chr, mat[n][:].T[0]))
                    .replace('/', '_')
                    .replace('.', '_'))[:-4]

            smisc.imsave(os.path.join(target_path[0 if c % train_images else 1],
                                      f'{name}-image.png'), img)
            smisc.imsave(os.path.join(target_path[0 if c % train_images else 1],
                                      f'{name}-depth.png'), depth)
            c += 1
            if c >= LIMIT:
                break


def main():
    path_train = os.path.join('data', 'train')
    path_test = os.path.join('data', 'test')
    try:
        os.makedirs(path_train, 0o755)
    except OSError:
        pass
    try:
        os.makedirs(path_test, 0o755)
    except OSError:
        pass

    print('\nPreprocessing data...')
    print(f'Images: {WIDTH}x{HEIGHT} Depths: {D_WIDTH}x{D_HEIGHT}')

    processors = {
        'make3d1': __process_make3d1,
        'make3d2': __process_make3d2,
        'nyu': __process_nyu,
    }

    for key, processor in processors.items():
        if key in sys.argv:
            print(f'Preprocessing {key}')
            processor(path_train, path_test)
    print('Preprocessing done.')


if __name__ == '__main__':
    main()
