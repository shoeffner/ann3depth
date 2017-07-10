# ANN for depth map estimations

Requires Python 3.6+.

`make`...
- `install`: Install python requirements.
- `datasets`: List available datasets.
- `download [key ...]`: Download datasets for keys. Use `key=all` to download all.
- `smart`: Create 1-page SMART goals presentation.
- `doc`: Compile documentation.

Examples:

```
make install
make datasets
make download make3d1
make download make3d2 nyu
make download all
make preprocess nyu
make smart
make doc
```


# Grid computations

To perform grid computations we use the Sun Grid Engine and anaconda.

## Setup

To setup the conda environment, run `make conda` on your grid root node. This
creates a conda environment named `asuckro-shoeffner-ann3depth` which installs
all packages. If you want to use a different environment name, always set the
environment variable `CONDAENV`.

## Learning

In the `tools/grid` directory you can find the `gridtrain.sge`. It's a
relatively generic file and uses mostly environment variables for parameter
adjustments.

You can start a grid job by running

```
make grid
```

You can change the following parameters:

- CONDAENV: the conda environment to use (important if you changed the setup step)
- NET: The network to train (e.g. `DownsampleNetwork`)
- EPOCHS: The number of epochs (Default: 500)
- BATCHSIZE: The number of images per batch (Default: 32)
- DATASETS: The datasets to learn (Default: make3d1)

An example run for the DownsampleNetwork could be:

```
NET=DownsampleNetwork BATCHSIZE=8 make grid
```
