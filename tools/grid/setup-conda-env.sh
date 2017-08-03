#!/usr/bin/env bash

CONDAENV=${CONDAENV-asuckro-shoeffner-ann3depth-env}

conda create -n $CONDAENV --yes python=3.6
source activate $CONDAENV

pip install --upgrade --ignore-installed -r requirements.txt
pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl
