# Advanced Deep Learning Seminar

This repository contains my weekly tasks, code, figures, and reports for an advanced deep learning seminar.

Each task is self-contained so that the corresponding code and PDF report can be found and understood quickly.

---

## Setup

Create and activate the environment:

```bash
mamba env create -f environment.yml
mamba activate deepLearning
```

(Optional) Add a Jupyter kernel:

```bash
python -m ipykernel install --user --name deepLearning
```

---

## Example Task

A complete working example is available in:

```
tasks/task_00_example
```

This task demonstrates the full workflow:

```
code → plots → report
```

If anything is unclear, this is the reference implementation.

---

## Usage

### Create a new task

```bash
make new NAME=task_01
```

### Generate plots

```bash
make plots TASK=task_01
```

* runs `code/make_plots.py`
* saves figures to `report/figures/`

### Compile report

```bash
make report TASK=task_01
```

* compiles LaTeX using `latexmk`
* writes temporary files to `build/`
* copies final PDF to:

```
report/main.pdf
```

### Run everything

```bash
make TASK=task_01
```

### Clean build files

```bash
make clean TASK=task_01
```

---

### Run inside a task

```bash
cd tasks/task_01
make
```

---

## Structure

```
tasks/
  task_01/
    code/      # scripts and notebooks
    report/    # LaTeX report + figures
    build/     # temporary files (auto-generated)
```

Workflow:

```
code → figures → PDF report
```

---

## Code Guidelines

* `code/make_plots.py` should stay **clean and minimal**

  * it is the *red line* of the task
  * shows clearly what is plotted and in which order

* more complex logic should be:

  * moved into helper scripts, or
  * explored in Jupyter notebooks

* notebooks (`.ipynb`) are useful for:

  * experimentation
  * prototyping
  * debugging

---

## Reports vs README

* The **report (`main.tex`)**:

  * explains methods
  * presents results
  * contains the main discussion

* The **task README**:

  * explains how to run the task
  * gives minimal context only

---

## Plotting & LaTeX

* plots are generated with matplotlib
* global style is defined in:

```
latex/matplotlibrc
```

* figures are saved to:

```
report/figures/
```

* included in LaTeX using:

```latex
\includegraphics{figures/plot.pdf}
```

* bibliography is shared via:

```
latex/references.bib
```

and included automatically through the header.

---

## Notes

* each task is independent
* shared configuration lives in `latex/`
* use `make` for a consistent and reproducible workflow
