from matplotlib import pyplot as plt

import data
import visualize


def main():
    print('loading data')
    dataset = data.training(['make3d1', 'make3d2'])
    visualize.DataBrowser(dataset)
    plt.show()


if __name__ == '__main__':
    main()
