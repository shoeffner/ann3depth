#!/usr/bin/env bash

CONDAENVGPU=${CONDAENV-ann3depth}-GPU
CONDAENVCPU=${CONDAENV-ann3depth}-CPU


conda create -n $CONDAENVGPU --yes python=3.6
source activate $CONDAENVGPU

pip install --upgrade --ignore-installed -r requirements.txt
pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl


conda create -n $CONDAENVCPU --yes python=3.6
source activate $CONDAENVCPU

pip install --upgrade --ignore-installed -r requirements.txt
pip install --upgrade --ignore-installed https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.2.1-cp36-cp36m-linux_x86_64.whl
