import argparse
import logging
import logging.config
import signal

import data
import networks


def test_network(*, network, datasets, samples, checkpoints,
                 tensorboard, **unused_catch_kwargs):
    dataset = data.test(dataset=datasets, samples=samples)
    network = network(dataset[0].img.shape, dataset[0].depth.shape,
                      ckptdir=checkpoints, tbdir=tensorboard, cont=True)
    network.test(dataset)

    import visualize
    visualize.Dataviewer(dataset, name='Results',
                         keys=['img', 'depth', 'result'],
                         cmaps={'depth': 'gray', 'result': 'gray'})


def train_network(*, network, epochs, batchsize, datasets, samples,
                  checkpoints, ckptfreq, tensorboard, cont,
                  **unused_catch_kwargs):
    dataset_train, dataset_test = data.load(dataset=datasets, samples=samples)
    network = network(dataset_train[0].img.shape, dataset_train[0].depth.shape,
                      ckptdir=checkpoints, ckptfreq=ckptfreq,
                      tbdir=tensorboard, cont=cont)
    network.train(dataset_train, dataset_test, epochs, batchsize)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('datasets', nargs='*',
                        default=['make3d1', 'make3d2'],
                        help='The datasets to use.')
    parser.add_argument('--samples', '-s', default='all',
                        help='Number of samples from dataset.')
    parser.add_argument('--train', '-t', action='store_const',
                        const=train_network, default=test_network,
                        help='If provided, the network is trained.')
    parser.add_argument('--network', '-n', default='DownsampleNetwork',
                        help='Enter a network name from networks.py.')
    parser.add_argument('--epochs', '-e', default=500, type=int,
                        help='Number of epochs')
    parser.add_argument('--batchsize', '-b', default=32, type=int,
                        help='Batchsize')
    parser.add_argument('--cont', '-c', action='store_true',
                        help='Continue from checkpoint.')
    parser.add_argument('--ckptdir', '-p', default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--ckptfreq', '-f', default=50, type=int,
                        help='Create a checkpoint every N epochs.')
    parser.add_argument('--tbdir', '-l', default='tb_logs',
                        help='Tensorboard directory')
    return parser.parse_args()


def maybe_int(v):
    try:
        return int(v)
    except ValueError:
        return v


if __name__ == '__main__':
    logging.config.fileConfig('logging.ini')
    logger = logging.getLogger('ann3depth')

    args = parse_args()
    logger.debug(args)

    timeout = 4200
    signal.alarm(timeout)
    logger.info(f'Killing job in {timeout} seconds.')

    args.train(network=getattr(networks, args.network),
               epochs=args.epochs,
               batchsize=args.batchsize,
               datasets=args.datasets,
               samples=maybe_int(args.samples),
               checkpoints=args.ckptdir,
               ckptfreq=args.ckptfreq,
               tensorboard=args.tbdir,
               cont=args.cont or (args.train == test_network)
               )
