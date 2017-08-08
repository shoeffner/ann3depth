# Default directory parameters
OUT_DIR := ./build
DATA_DIR ?= ./data
LOG_DIR := ./grid_logs
TB_PORT ?= 5003

# Grid parameters
CONDAENV ?= asuckro-shoeffner-ann3depth
PS_NODES ?= 0
WORKERS ?= 1

ifdef CLUSTER_SPEC
	CLUSTER_PARAM1 = --cluster-spec=${CLUSTER_SPEC}
else
	CLUSTER_PARAM1 :=
endif
ifdef JOB_TYPE
	CLUSTER_PARAM2 = --job-name=${JOB_TYPE}
else
	CLUSTER_PARAM2 :=
endif
ifdef TASK_INDEX
	CLUSTER_PARAM3 = --task-index=${TASK_INDEX}
else
	CLUSTER_PARAM3 :=
endif
CLUSTER_PARAMS ?= ${CLUSTER_PARAM1} ${CLUSTER_PARAM2} ${CLUSTER_PARAM3}

# Default training parameters
MODEL ?= dcnf
STEPS ?= 10000000
BATCHSIZE ?= 32
DATASET ?= nyu
SUM_FREQ ?= 300
CKPT_FREQ ?= 900
CKPT_DIR := checkpoints
TIMEOUT ?= 4200
CONT ?=

# Preprocessing parameters
WIDTH ?= 640
HEIGHT ?= 480
DHEIGHT ?= 55
DWIDTH ?= 73
FORCE ?=
START ?= 0
LIMIT ?=

SCRIPT := python3 src/ann3depth.py
COMMON_PARAMETERS := --ckptdir=${CKPT_DIR} --datadir=${DATA_DIR} --model=${MODEL} ${CLUSTER_PARAMS}
TRAIN_PARAMETERS := --steps=${STEPS} --batchsize=${BATCHSIZE} --ckptfreq=${CKPT_FREQ} --sumfreq=${SUM_FREQ} --timeout=${TIMEOUT} ${CONT}

# Check if download is wanted, and if so, set dataset names
# see http://stackoverflow.com/a/14061796/3004221
ifeq (download,$(firstword $(MAKECMDGOALS)))
    DATASET := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
    $(eval $(DATASET):;@:)
endif

# Check if preprocessing is wanted, and if so, set dataset names
ifeq (preprocess,$(firstword $(MAKECMDGOALS)))
    DATASET := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
    $(eval $(DATASET):;@:)
endif

# Check if convert is wanted, and if so, set dataset names
ifeq (convert,$(firstword $(MAKECMDGOALS)))
    DATASET := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS))
    $(eval $(DATASET):;@:)
endif


####### TRAINING ##########

# Trains the network
.PHONY: train
train: ${DATA_DIR}
	${SCRIPT} ${COMMON_PARAMETERS} ${TRAIN_PARAMETERS} ${DATASET}

# Reloads a checkpoint and continues training
.PHONY: continue
continue: data
	${SCRIPT} --cont ${COMMON_PARAMETERS} ${TRAIN_PARAMETERS} ${DATASET}

# Submit a grid training job
.DEFAULT: distributed
.PHONY: distributed
distributed: ${LOG_DIR} ./tools/grid/startup.sh
	PS_NODES=${PS_NODES} WORKERS=${WORKERS} CONDAENV=${CONDAENV} MODEL=${MODEL} STEPS=${STEPS} BATCHSIZE=${BATCHSIZE} DATASET=${DATASET} CKPT_DIR=${CKPT_DIR} CKPT_FREQ=${CKPT_FREQ} SUM_FREQ=${SUM_FREQ} DATA_DIR=${DATA_DIR} TIMEOUT=${TIMEOUT} CONT=${CONT} qsub ./tools/grid/distributed_master.sge



####### HELPER ##########

.PHONY: help
help:
	${SCRIPT} --help

# Installs the requirements from the requirements file
.PHONY: install
install: ./tools/grid/startup.sh requirements.txt
	pip3 install -r requirements.txt -U

# list datasets to be used with download target
.PHONY: datasets
datasets:
	python3 tools/data_downloader.py --list

# download data sets and extract them
.PHONY: download
download: ${DATA_DIR}
	DATA_DIR=${DATA_DIR} python3 tools/data_downloader.py $(DATASET)

# preprocess data sets and extract them
.PHONY: preprocess
preprocess: ${DATA_DIR}
	DATA_DIR=${DATA_DIR} WIDTH=${WIDTH} HEIGHT=${HEIGHT} DHEIGHT=${DHEIGHT} DWIDTH=${DWIDTH} FORCE=${FORCE} START=${START} LIMIT=${LIMIT} python3 tools/data_preprocessor.py $(DATASET)

# convert to tf records
.PHONY: convert
convert: ${DATA_DIR}
	DATA_DIR=${DATA_DIR} python3 tools/data_tf_converter.py $(DATASET) --del_raw

# Opens up tensorboard for inspection of graphs and summaries
.PHONY: tb
tb: ${CKPT_DIR}
	tensorboard --logdir=${CKPT_DIR} --host=0.0.0.0 --port=${TB_PORT}

# Create conda environment used by grid computation servers
.PHONY: conda
conda: ./tools/grid/startup.sh
	CONDAENV=${CONDAENV} /bin/bash ./tools/grid/setup-conda-env.sh

# Create the startup.sh which will be sourced by the grid master
./tools/grid/startup.sh: ./tools/configure.py
	python3 ./tools/configure.py


####### DOCUMENTATION ##########

.PHONY: doc
doc: ${OUT_DIR}
	pandoc \
		-o ${OUT_DIR}/anndepth_assh_documentation.pdf \
		--bibliography=docs/references.bib \
		docs/documentation.md \
		docs/documentation.yaml


# 1 page SMART goals presentation slide
.PHONY: smart
smart: ${OUT_DIR}
	pandoc -t beamer \
		-o ${OUT_DIR}/anndepth_assh_smart.pdf \
		docs/presentations/SMART-presentation.md \
		docs/presentations/SMART-presentation.yaml

# Status presentation slides
.PHONY: status
status: ${OUT_DIR}
	pandoc -t beamer \
		-o ${OUT_DIR}/anndepth_assh_status.pdf \
		docs/presentations/Status-presentation.md \
		docs/presentations/Status-presentation.yaml



####### UTILITY ##########

${OUT_DIR}:
	@mkdir -p ${OUT_DIR}

${DATA_DIR}:
	@mkdir -p ${DATA_DIR}

${LOG_DIR}:
	@mkdir -p ${LOG_DIR}

${CKPT_DIR}:
	@mkdir -p ${CKPT_DIR}
