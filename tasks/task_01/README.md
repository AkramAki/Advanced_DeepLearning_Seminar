# Running the Tasks

Make sure the correct conda/mamba environment is activated before running anything:

```bash
mamba activate deepLearning
```

The environment installs the repository in editable mode (`-e .`). This is required so imports from the `common/` folder work correctly.

## Important

Before building the report, you must run the notebook for the task once.

Example for Task 01:

```text
tasks/task_01/code/task_01.ipynb
```

When the notebook is executed completely:

* the dataset is checked and downloaded automatically if missing (~1 GB)
* the CNN model is trained
* plots and figures for the report are generated
* the trained model is saved into the `models/` folder

The first full execution may take a while.

After the model has been trained once, search the notebook for:

```python
skip_training
```

and set it to `True`.

This skips retraining and loads the saved model from the `models/` folder instead, which makes rerunning the notebook much faster.

The dataset is only downloaded once and reused afterwards.

## Data Setup

When running the notebook for this task, the data is downloaded automatically.

* If the data already exists in:

```text
data/task_01/
```

the download step is skipped.

* Otherwise, the dataset is downloaded and stored there.

After a successful download, the directory should look like:

```text
data/task_01/
  .cache/
  labels.npy
  spectra.npy
```

## Building the Report

After the notebook has been executed once, build the report from the repository root:

```bash
make TASK=task_01
```

This builds the report automatically using the generated figures and outputs the final PDF.
