# Default directory parameters
OUT_DIR := build
DATA_DIR := data
CKPT_DIR := checkpoints
TB_DIR := tb_logs
LOG_DIR := grid_logs

# Grid parameters
CONDAENV ?= asuckro-shoeffner-ann3depth

# Default training parameters
NET ?= DownsampleNetwork
EPOCHS ?= 500
BATCHSIZE ?= 32
DATASETS ?= make3d1
CKPT_FREQ ?= 50
CONT ?=

SCRIPT := python3 src/ann3depth.py
COMMON_PARAMETERS := --ckptdir=${CKPT_DIR} --tbdir=${TB_DIR} --network=${NET}
TRAIN_PARAMETERS := --epochs=${EPOCHS} --batchsize=${BATCHSIZE} --ckptfreq=${CKPT_FREQ} ${CONT}

# Check if download is wanted, and if so, set dataset names
# see http://stackoverflow.com/a/14061796/3004221
ifeq (download,$(firstword $(MAKECMDGOALS)))
    DATASETS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
    $(eval $(DATASETS):;@:)
endif

# Check if preprocessing is wanted, and if so, set dataset names
ifeq (preprocess,$(firstword $(MAKECMDGOALS)))
    DATASETS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
    $(eval $(DATASETS):;@:)
endif

# Makes a forward pass to inspect results of a trained network
.PHONY: inspect
inspect: data
	${SCRIPT} ${COMMON_PARAMETERS} ${DATASETS}

# Trains the network
.PHONY: train
train: data
	${SCRIPT} --train ${COMMON_PARAMETERS} ${TRAIN_PARAMETERS} ${DATASETS}

# Reloads a checkpoint and continues training
.PHONY: continue
continue: data
	${SCRIPT} --train --cont ${COMMON_PARAMETERS} ${TRAIN_PARAMETERS} ${DATASETS}

# Create conda environment used by grid computation servers
.PHONY: conda
conda:
	CONDAENV=${CONDAENV} /bin/bash ./tools/grid/setup-conda-env.sh

# Submit a grid training job
.PHONY: grid
grid: ${LOG_DIR}
	CONDAENV=${CONDAENV} NET=${NET} EPOCHS=${EPOCHS} BATCHSIZE=${BATCHSIZE} DATASETS=${DATASETS} CKPT_FREQ=${CKPT_FREQ} qsub ./tools/grid/train.sge

.PHONY: help
help:
	${SCRIPT} --help

# inspect samples
.PHONY: browse
browse:
	PYTHONPATH=src python3 src/visualize/dataviewer.py

# project documentation
.PHONY: doc
doc: ${OUT_DIR}
	pandoc \
		-o ${OUT_DIR}/anndepth_assh_documentation.pdf \
		--bibliography=docs/references.bib \
		docs/documentation.md \
		docs/documentation.yaml


# list datasets to be used with download target
.PHONY: datasets
datasets:
	@python3 tools/data_downloader.py --list


# download data sets and extract them
.PHONY: download
download: ${DATA_DIR}
	@python3 tools/data_downloader.py $(DATASETS)


# preprocess data sets and extract them
.PHONY: preprocess
preprocess: download ${DATA_DIR}
	@python3 tools/data_preprocessor.py $(DATASETS)


# 1 page SMART goals presentation slide
.PHONY: smart
smart: ${OUT_DIR}
	pandoc -t beamer \
		-o ${OUT_DIR}/anndepth_assh_smart.pdf \
		docs/SMART-presentation.md \
		docs/SMART-presentation.yaml

# Opens up tensorboard for inspection of graphs and summaries
.PHONY: tb
tb:
	tensorboard --logdir=${TB_DIR}

# Installs the requirements from the requirements file
.PHONY: install
install: requirements.txt
	pip3 install -r requirements.txt -U

${OUT_DIR}:
	@mkdir -p ${OUT_DIR}

${DATA_DIR}:
	@mkdir -p ${DATA_DIR}

${LOG_DIR}:
	@mkdir -p ${LOG_DIR}
