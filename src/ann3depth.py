from matplotlib import pyplot as plt

import data
import visualize
import networks
import threading


def train_network(network, dataset, epochs):
    network(dataset)


def browse_data(dataset):
    visualize.DataBrowser(dataset, key='depth')
    plt.show(False)


def browse_results(dataset):
    visualize.DataBrowser(dataset, key='result', cmap='gray')
    plt.show(True)


def generate_network(network, dataset):
    return network(dataset[0].img.shape, dataset[0].depth.shape)


def main():
    print('loading data')
    dataset = data.training(dataset=['make3d1', 'make3d2'], samples=20)

    network = generate_network(networks.FlatNetwork, dataset)

    training = threading.Thread(target=train_network,
                                args=(network, dataset, 1))
    training.start()

    browse_data(dataset)

    training.join()

    browse_results(dataset)


if __name__ == '__main__':
    main()
