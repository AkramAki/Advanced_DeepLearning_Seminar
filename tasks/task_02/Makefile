.PHONY: all clean

BUILD_DIR := build

# final target
all: report/main.pdf

# create folders
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# figures (only rerun if script changes or files missing)
build/figures.pdf: # code/plot.py | $(BUILD_DIR)
	echo "No automatic figure generation implemented yet. Please run the plotting script manually to generate the figures."
# PYTHONPATH=../.. MPLCONFIGDIR=$(BUILD_DIR) python code/plot.py 

# report (reruns only if tex or figures changed)
report/main.pdf: report/main.tex build/figures.pdf
	cd report && latexmk -lualatex -output-directory=../$(BUILD_DIR) main.tex
	cp build/main.pdf report/main.pdf

# clean everything
clean:
	rm -rf $(BUILD_DIR)
	rm -f report/main.pdf