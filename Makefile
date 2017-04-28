OUT_DIR := build

smart: ${OUT_DIR}
	pandoc -t beamer \
		-o ${OUT_DIR}/anndepth_assh_smart.pdf \
		docs/SMART-presentation.md \
		docs/SMART-presentation.yaml

${OUT_DIR}:
	mkdir -p ${OUT_DIR}
