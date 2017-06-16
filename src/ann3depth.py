from matplotlib import pyplot as plt

import data
import visualize
import networks
import threading


def train_network(network, dataset, epochs):
    print(network, len(dataset), epochs)


def browse_data(dataset, blocking=True):
    visualize.DataBrowser(dataset)
    plt.show(blocking)


def generate_network(network, dataset):
    return network(dataset[0].img.shape, dataset[0].depth.shape)


def main():
    print('loading data')
    dataset = data.training(dataset=['make3d1', 'make3d2'], samples=20)

    network = generate_network(networks.FlatNetwork, dataset)

    training = threading.Thread(target=train_network,
                                args=(network, dataset, 1))
    training.start()

    browse_data(dataset, blocking=False)

    training.join()

    results = dataset

    browse_data(results)


if __name__ == '__main__':
    main()
