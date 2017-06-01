OUT_DIR := build
DATA_DIR := data

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

.PHONY: install
install: requirements.txt
	pip install -r requirements.txt -U

${OUT_DIR}:
	@mkdir -p ${OUT_DIR}

${DATA_DIR}:
	@mkdir -p ${DATA_DIR}
