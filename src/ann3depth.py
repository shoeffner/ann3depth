import argparse
import os
import threading

from matplotlib import pyplot as plt

import data
import visualize
import networks


def generate_network(network, dataset):
    return network(dataset[0].img.shape, dataset[0].depth.shape)


def train_network(network, dataset, epochs, batchsize):
    network(dataset, epochs, batchsize)


def browse_data(dataset):
    visualize.Dataviewer(dataset,
                         name='Inputs',
                         keys=['img', 'depth'],
                         cmaps={'depth': 'gray'})


def browse_results(dataset):
    visualize.Dataviewer(dataset,
                         name='Results',
                         keys=['img', 'depth', 'result'],
                         cmaps={'depth': 'gray', 'result': 'gray'})


def main(browse=False, show_results=False):
    samples = 'all'
    datasets = ['make3d1', 'make3d2']
    print(f'Loading {samples} samples from {datasets}.')
    dataset = data.training(dataset=datasets, samples=samples)

    # Open data browser if requested
    if browse:
        print('Opening dataviewer for inputs (only first 20 samples)')
        browse_data(dataset[:30].copy())
        plt.show(False)

    network = generate_network(networks.DepthMapNetwork, dataset)

    epochs = 500
    batchsize = 32
    training = threading.Thread(target=train_network,
                                args=(network, dataset, epochs, batchsize))
    print(f'Beginning training of {epochs} epochs, batchsize: {batchsize}.')
    training.start()
    training.join()
    print('Training done.')

    # Open results browser if requested
    if show_results:
        print('Opening dataviewer for results (only first 20 samples)')
        browse_results(dataset[:30])
        plt.show(True)
    elif browse:
        plt.show(True)


def ensure_envs():
    if 'CKPT_DIR' not in os.environ:  # Checkpoint directory
        os.environ['CKPT_DIR'] = 'checkpoints'
    if 'TB_DIR' not in os.environ:  # TensorBoard directory
        os.environ['TB_DIR'] = 'tb_logs'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help='Enable data viewer')
    parser.add_argument('-r', action='store_true', help='Enable result viewer')
    args = parser.parse_args()

    ensure_envs()

    main(browse=args.d, show_results=args.r)
