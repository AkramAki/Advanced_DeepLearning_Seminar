.PHONY: all clean

BUILD_DIR := build

# final target
all: report/main.pdf

# create folders
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# figures (only rerun if script changes or files missing)
build/figures.pdf: code/make_plots.py | $(BUILD_DIR)
	PYTHONPATH=../.. MPLCONFIGDIR=$(BUILD_DIR) python code/make_plots.py 

# report (reruns only if tex or figures changed)
report/main.pdf: report/main.tex build/figures.pdf
	cd report && latexmk -lualatex -output-directory=../$(BUILD_DIR) main.tex
	cp build/main.pdf report/main.pdf

# clean everything
clean:
	rm -rf $(BUILD_DIR)
	rm -f report/main.pdf