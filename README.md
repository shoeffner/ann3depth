# ANN for depth map estimations

This project originally aimed at creating depth maps for single monocular
images. During the process of the development it became a study on how to run
TensorFlow distributed on the Sun Grid Engine to achieve computations of deep
networks.

To learn distributed computing we implemented the Distributed Convolutional
Neural Fields by Liu et al. (2015) (DOI:
[10.1109/TPAMI.2015.2505283](https://doi.org/10.1109/TPAMI.2015.2505283)) and
the Multi-Scale Deep Network by Eigen et al. (2014) (arxiv:
[arXiv:1406.2283](https://arxiv.org/abs/1406.2283)).

However, since we didn't yet manage to really make everything distributed
(especially the bigger matrix operations), we tuned down the DCNF network and
added some assumptions, see [below](#changes-to-liu-et-al). It worked better to
the MSDN approach, but we also made some [changes](#changes-to-eigen-et-al).


## Set up

Requires Python 3.6+.

1. Run `make install` to install the python requirements.
2. If you want to run distributed training, run `make conda` to setup the
   proper anaconda environments.
3. If you don't want the data to be stored in ./data, set the environment
   variable `DATA_DIR`: `export DATA_DIR=./your/path/here`.
4. Run `make download make3d1 make3d2 nyu`, followed by `make preprocess
   make3d1 make3d2 nyu`. Check below or the Makefile for parameters for the
   preprocessing.
5. If you don't need the original files you can just run `make convert make3d1
   make3d2 nyu` to convert all data to
   [TFRecords](https://www.tensorflow.org/api_guides/python/python_io). If you
   still need the originals, make a copy of them.


## Training

### Distributed training

*Note*: For the distributed TensorFlow jobs your machines are assumed to:
a) have access to the same hard drives to share the cluster specs created by
the distributed master
b) have ports 5001 and 5002 open so that grid jobs can communicate over them.
This is an "as-is" project as so many out there and we didn't really built more
elaborate solutions. However, you can change the ports in
[tools/grid/split_resources.py](tools/grid/split_resources.py).

For distributed training, just run `make distributed`. It will select the
`msdn` model, the default dataset (`nyu`) and train on the Sun Grid Engine.

The `dcnf` model needs at least four workers, so you should instead do a
call like this:

```bash
WORKERS=4 PS_NODES=8 MODEL=dcnf make distributed
```

To browse the TensorBoard (on [http://localhost:5003](http://localhost:5003)),
use the target `tb`:

```bash
make tb
```

You can also change the TensorBoard port by additionally supplying `TB_PORT=...`.


### Local training

You can just run `make train` with the same parameters as `make distributed` to
train locally. This is very useful for debugging.


### Test and validation

So far we have not implemented a test or validation yet. This is because the
training is not working well yet.


### Makefile targets

These are the Makefile targets you can use.

- `help`: Prints the help for the training script.
- `install`: sets up the environment.
- `datasets`: Lists all available datasets.
- `download [dataset, ...]`: Downloads the specified datasets.
- `preprocess [dataset, ...]`: Preprocesses the datasets.
- `convert [dataset, ...]`: Converts the datasets.
- `tb`: Runs TensorBoard (blocking call).
- `conda`: Sets up the grid conda environments.
- `doc`: Creates the documentation report.
- `smart`: Creates the project proposal presentation.
- `status`: Creates the status report we gave at the end of the seminar, but
  before really having implemented much.


### Parameters

These are the parameters you can set for the Makefile. They all have some
useful defaults, so that you could potentially just run `make distributed` or
`make train` and would be fine. For more advanced experiments check the
parameters below.

#### Preprocessing

- `WIDTH`: The image width to rescale to (default `640`).
- `HEIGHT`: The image height to rescale to (default `480`).
- `DWIDTH`: The target (depth) width to rescale to (default `55`).
- `DHEIGHT`: The target (depth) width to rescale to (default `73`).
- `FORCE`: Overwrites existing preprocessed data (default `False`).
- `START`: The image to start with (default `0`).
- `LIMIT`: The maximum number of images to process. If empty (default), all
  images (from the start on) are processed.

#### Training

- `MODEL`: The model to use. Allowed are `msdn` (Eigen et al., default) and
  `dcnf` (Liu et al.)
- `RUNID`: The suffix to distinguish checkpoint paths, e.g. the experiment or
  setup identifier.
- `STEPS`: The maximum number of steps to train (1 step = 1 batch, default
  `10000000`). Note that the `msdn` model already stops after a couple 100k
  steps to train.
- `BATCHSIZE`: The batch size (default `32`).
- `DATASET`: The dataset to train on (default `nyu`).
- `SUM_FREQ`: The summary frequency in steps (default `100`).
- `CKPT_FREQ`: The checkpoint frequency in seconds (default `900`).
- `CKPT_DIR`: The checkpoint directory (default `checkpoints`).
- `TIMEOUT`: The time (in seconds) after which the jobs should kill themselves
  (default `4200`).

#### Other grid parameters

- `CONDAENV`: The conda environment prefix to use (default `ann3depth`). The
  scripts append `-GPU` and `-CPU` to select the proper environments.
- `PS_NODES`: The number of parameter server nodes (default `1`).
- `WORKERS`: The number of worker nodes (default `0`).
- `GRID_PREFIX`: The prefix to distinguish different jobs on the SGE. Used for
  the keepalive jobs as well as to visually distinguish jobs when running
  `qstatus` (default `a3dx`).
- `MEMORY_RATIO`: The amount of memory to reserve on target machines in parts
  of the available memory (default `0.3`).

#### Other parameters

- `DATA_DIR`: The directory to store the data and retrieve it from (default `./data`).
- `TB_PORT`: The port to serve TensorBoard at (default `5003`).


## Changes to Liu et al.

These are the changes we made to get the network running and make it distributed:

* Smaller input images: Even though it's not mentioned in their paper, Liu et
  al. seem to use bigger images than we do, ours are just 240x320 pixels big.
* Smaller image patches: Instead of 224x224 we use 100x100 patches.
* No superpixels: Since superpixels are not really nice for the tensorflow
  pipeline, we started using only small image pixels of 40x40 pixels and
  haven't gone to the other approach yet.
* Removal of first dense layer in unary part: The layer with 4096 neurons was
  just too big for all GPUs we had available.
* We use the mean loss over a batch instead of the loss for each image: In
  practice this doesn't make much of a difference as we just have one batch
  item anyways.
* Only two difference measures in the pairwise part: We did not implement the
  texture differences, but the histogram and color differences. However, it was
  unclear how the histogram worked in Liu et al., so it is possible that our
  implementation differs wildly.
* Fixed gamma to 1 (not reported in original paper).


## Changes to Eigen et al.

* No distorted versions used for training.
* Hardware setup changed.
* Step counts are handled differently, resulting in possibly slightly different counts.
* Results indicate that the wrong loss is applied to the coarse layers, but
  this needs further investigation.


## References

1. Fayao Liu, Chunhua Shen, Guosheng Lin, and Ian Reid: Learning Depth from
   Single Monocular Images Using Deep Convolutional Neural Fields. IEEE PAMI,
   2015.
2. Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus: Indoor
   Segmentation and Support Inference from RGBD Images. ECCV, 2012.
3. Ashutosh Saxena, Sung H. Chung and Andrew Y. Ng: Learning Depth from Single
   Monocular Images. NIPS, 2005.
4. Ashutosh Saxena, Sung H. Chung and Andrew Y. Ng: 3-D Depth Reconstruction
   from a Single Still Image. IJCV, 2007.
4. Ashutosh Saxena, Min Sun and Andrew Y. Ng: Make3D: Learning 3D Scene
   Structure from a Single Still Image. IEEE PAMI, 2009.
5. David Eigen, Christian Puhrsch, and Rob Fergus: Depth Map Prediction from a
   Single Image using a Multi-Scale Deep Network. NIPS, 2014.
