# ANN for depth map estimations

This project originally aimed at creating depth maps for single monocular images. During the process of the development it became a study on how to run TensorFlow distributed on the Sun Grid Engine to achieve computations of deep networks.


## Set up

Requires Python 3.6+.

1. Run `make install` to install the python requirements.
2. If you don't want the data to be stored in ./data, set the environment variable DATA\_DIR: `export DATA_DIR=./your/path/here`.
3. Run `make download make3d1 make3d2 nyu`, followed by `make preprocess make3d1 make3d2 nyu`. Check below (or the Makefile) for parameters for the preprocessing.
4. If you don't need the original files you can just run `make convert make3d1 make3d2 nyu` to convert all data to [TFRecords](https://www.tensorflow.org/api_guides/python/python_io). If you still need the originals, make a copy of them. (This might change in the future.)

## Training

### General training

```
usage: ann3depth.py [-h] [--model MODEL] [--steps STEPS]
                    [--batchsize BATCHSIZE] [--ckptdir CKPTDIR]
                    [--ckptfreq CKPTFREQ] [--datadir DATADIR] [--cont]
                    [--timeout TIMEOUT] [--cluster-spec CLUSTER_SPEC]
                    [--job-name JOB_NAME] [--task-index TASK_INDEX]
                    [datasets [datasets ...]]

positional arguments:
  datasets              The datasets to use.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL, -m MODEL
                        Enter a model name.
  --steps STEPS, -s STEPS
                        Total steps
  --batchsize BATCHSIZE, -b BATCHSIZE
                        Batchsize
  --ckptdir CKPTDIR, -p CKPTDIR
                        Checkpoint directory
  --ckptfreq CKPTFREQ, -f CKPTFREQ
                        Create a checkpoint every N seconds.
  --datadir DATADIR, -d DATADIR
                        The data directory containing the datasets.
  --cont, -c            Continue from checkpoint
  --timeout TIMEOUT, -k TIMEOUT
                        The time after which the process dies.
  --cluster-spec CLUSTER_SPEC
                        The path to the cluster specification json.
  --job-name JOB_NAME   "worker" or "ps" for distributed computations.
  --task-index TASK_INDEX
                        Task index for distributed computations.
```


### Local training


### Distributed training

