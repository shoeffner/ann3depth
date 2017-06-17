import glob
import os
import random

import scipy.misc as smisc


class Sample:
    """Represents a sample with name, data, etc."""

    def __init__(self, img_path, depth_path):
        self.name = os.path.splitext(os.path.basename(depth_path))[0][:-6]
        self.dataset = os.path.basename(os.path.dirname(depth_path))
        self.img = smisc.imread(img_path)
        self.depth = smisc.imread(depth_path)
        self.result = None

    def __str__(self):
        return '\n  '.join(
            [f'Sample: {self.name}, Set: {self.dataset}',
             f'Depth - shape: {self.depth.shape}, dtype: {self.depth.dtype}',
             f'Image - shape: {self.img.shape}, dtype: {self.img.dtype}'])


def __get_files(subdir, dataset):
    """Reads all filepaths in data/{subdir}/{dataset}, separated by -depth and
    -image suffixes.

    Args:
        subdir: The directory (train or test) to search.
        dataset: The dataset to search (subdir of subdir).

    Returns:
        A list of the form: [(depth_path, img_path), ...]
    """
    if isinstance(dataset, str):
        depths = glob.glob(os.path.join('data', subdir,
                                        '*' if dataset == 'all' else dataset,
                                        '*-depth.png'))
        images = glob.glob(os.path.join('data', subdir,
                                        '*' if dataset == 'all' else dataset,
                                        '*-image.png'))
    else:
        depths = []
        images = []
        for ds in dataset:
            depths += glob.glob(os.path.join('data', subdir, ds, '*-depth.png'))
            images += glob.glob(os.path.join('data', subdir, ds, '*-image.png'))

    return list(zip(images, depths))


def __sample(sample_list, samples):
    """Samples the sample_list.

    If samples is 'all', all samples are returned (this becomes the identity
    function).

    If samples is an integer, the number of samples is drawn
    (without replacement) from the given list.

    If samples is a list of integers, they are used for indexing.

    Args:
        sample_list: The list of samples to draw from.
        samples: see above, either 'all', a number or a list.

    Returns:
        Samples from the list.
    """
    if samples == 'all':
        return sample_list

    if isinstance(samples, int):
        return random.sample(sample_list, samples)

    max = len(sample_list)
    return [sample_list[sample] for sample in samples if 0 <= sample < max]


def __load(subdir, *, samples='all', dataset='all'):
    """Loads data from a given subdir.

    Args:
        samples: See __sample for details.
        dataset: See __get_files for details.

    Returns:
        A list of Samples.
    """
    files = __sample(__get_files(subdir, dataset), samples)
    return [Sample(i, d) for i, d in files]


def training(*, samples='all', dataset='all'):
    """Loads training samples.

    Args:
        samples: See __sample for details.
        dataset: See __get_files for details.

    Returns:
        A list of Samples.
    """
    return __load('train', samples=samples, dataset=dataset)


def test(*, samples='all', dataset='all'):
    """Loads test samples.

    Args:
        samples: See __sample for details.
        dataset: See __get_files for details.

    Returns:
        A list of Samples.
    """
    return __load('test', samples=samples, dataset=dataset)


def mixed(*, samples='all', dataset='all'):
    """Loads mixed samples (test and training). Mostly useful for data
    inspection.

    Args:
        samples: See __sample for details.
        dataset: See __get_files for details.

    Returns:
        A list of Samples.
    """
    return __load('*', samples=samples, dataset=dataset)


def load(*, samples='all', dataset='all'):
    """Loads training and test samples.

    Args:
        samples: See __sample for details.
        dataset: See __get_files for details.

    Returns:
        A list of training Samples and a list of test Samples.
    """
    return (training(samples=samples, dataset=dataset),
            test(samples=samples, dataset=dataset))
