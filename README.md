# ANN for depth map estimations

This project originally aimed at creating depth maps for single monocular
images. During the process of the development it became a study on how to run
TensorFlow distributed on the Sun Grid Engine to achieve computations of deep
networks.

To learn distributed computing we implement the Distributed Convolutional
Neural Fields by Liu et al. (2015) (DOI:
[10.1109/TPAMI.2015.2505283](https://doi.org/10.1109/TPAMI.2015.2505283)).

However, since we didn't yet manage to really make everything distributed
(especially the bigger matrix operations), we tuned down the network and added
some assumptions. (See more [below](#Changes-to-Liu-et-al.)


## Set up

Requires Python 3.6+.

1. Run `make install` to install the python requirements.
2. If you don't want the data to be stored in ./data, set the environment variable `DATA_DIR`: `export DATA_DIR=./your/path/here`.
3. Run `make download make3d1 make3d2 nyu`, followed by `make preprocess make3d1 make3d2 nyu`. Check below (or the Makefile) for parameters for the preprocessing.
4. If you don't need the original files you can just run `make convert make3d1 make3d2 nyu` to convert all data to [TFRecords](https://www.tensorflow.org/api_guides/python/python_io). If you still need the originals, make a copy of them.


## Training

### Distributed training

*Note*: It is currently not supported to run multiple grid jobs with different
tasks next to these for the same user! Be careful, as the
`distributed_master.sge` might kill your jobs!

*Note*: For the distributed TensorFlow jobs your machines are assumed to:
a) have access to the same hard drives to share the cluster specs created by
the distributed master
b) have ports 5001 and 5002 open so that grid jobs can communicate over them.
This is an "as-is" project as so many out there and we didn't really build more
elaborate solutions. You can however change the ports in [tools/grid/split_resources.py](tools/grid/split_resources.py).

For distributed training, just run `make distributed`. It will select the
`dcnf` model, the default dataset (`nyu`) and train on the Sun Grid Engine.
However, that model needs at least four workers, so you should instead do a
call like this:

```bash
WORKERS=4 PS_NODES=8 make distributed
```

A common idea is to also remove the old checkpoints when kicking off a new run
(as well as cleaning up grid logs) and starting TensorBoard automatically, so a
very common command to get everything up and running is:

```bash
(qdel -u $(whoami) && sleep 2 || true) && \
   rm -rf checkpoints/dcnf && rm -rf grid_logs && \
   BATCHSIZE=1 WORKERS=4 PS_NODES=8 make distributed tb
```

This deletes your user's current grid jobs, waits until they are cleared from
the queues, removes the checkpoints for the dcnf model, removes the gridlogs
and starts the training with a batch size of 1, 4 worker nodes, 8 parameter
servers and allows to browse the TensorBoard (on [http://localhost:5003](http://localhost:5003)).

You can also change the TensorBoard port by additionally supplying `TB_PORT=...`.


### Local training

At the moment, local training is in theory support for freshly written models,
but the only serious implementation relies on some manual placement on
different workers, which makes it impossible to run it locally.


### Test and validation

So far we have not implemented a test or validation yet. This is because the
training is not working well yet.


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
