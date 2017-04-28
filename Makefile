OUT_DIR := build

doc: ${OUT_DIR}
	pandoc \
		-o ${OUT_DIR}/anndepth_assh_documentation.pdf \
		--bibliography=docs/references.bib \
		docs/documentation.md \
		docs/documentation.yaml

smart: ${OUT_DIR}
	pandoc -t beamer \
		-o ${OUT_DIR}/anndepth_assh_smart.pdf \
		docs/SMART-presentation.md \
		docs/SMART-presentation.yaml

${OUT_DIR}:
	@mkdir -p ${OUT_DIR}
