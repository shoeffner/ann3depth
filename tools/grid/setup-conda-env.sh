#!/usr/bin/env bash

CONDAENVGPU=${CONDAENV-ann3depth}-GPU
CONDAENVCPU=${CONDAENV-ann3depth}-CPU


conda create -n $CONDAENVGPU --yes python=3.6
source activate $CONDAENVGPU
conda info --envs | grep '*'

pip install --upgrade -r requirements.txt -r requirements-gpu.txt


conda create -n $CONDAENVCPU --yes python=3.6
source activate $CONDAENVCPU
conda info --envs | grep '*'

pip install --upgrade -r requirements.txt -r requirements-cpu.txt
