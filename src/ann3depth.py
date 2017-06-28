import argparse

from matplotlib import pyplot as plt

import data
import visualize
import networks
import threading


def generate_network(network, dataset):
    return network(dataset[0].img.shape, dataset[0].depth.shape)


def train_network(network, dataset, epochs):
    network(dataset)


def browse_data(dataset):
    visualize.DataBrowser(dataset,
                          name='Inputs',
                          keys=['img', 'depth'],
                          cmaps={'depth': 'jet'})


def browse_results(dataset):
    visualize.DataBrowser(dataset,
                          name='Results',
                          keys=['img', 'depth', 'result'],
                          cmaps={'depth': 'jet', 'result': 'gray'})


def main(browse=False, show_results=False):
    dataset = data.training(dataset=['make3d1', 'make3d2'], samples=20)

    network = generate_network(networks.DepthMapNetwork, dataset)

    training = threading.Thread(target=train_network,
                                args=(network, dataset, 1))
    training.start()

    # Open data browser if requested
    if browse:
        browse_data(dataset)
        if show_results:
            plt.show(False)

    training.join()

    # Open results browser if requested
    if show_results:
        browse_results(dataset)
        plt.show(True)
    elif browse:
        plt.show(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', action='store_true', help='Enable data viewer')
    parser.add_argument('-r', action='store_true', help='Enable result viewer')
    args = parser.parse_args()

    main(browse=args.d, show_results=args.r)
