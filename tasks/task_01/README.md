# Running Task 02

Make sure the correct conda/mamba environment is active before running anything:

```bash
mamba activate deepLearning
```

The environment installs the repository in editable mode (`-e .`). This is required so imports from the `common/` folder work correctly.

## Important

Before building the report, the notebook for Task 02 has to be run once:

```text
tasks/task_02/code/task_02.ipynb
```

When the notebook is executed completely:

* the dataset is checked and downloaded automatically if missing
* the uncertainty-aware CNN models are trained
* plots and figures for the report are generated
* the trained models are saved into the `models/` folder

The first full execution may take a while because all model configurations have to be trained.

After the models have been trained once, search the notebook for:

```python
skip_training
```

and set it to `True`.

This skips retraining and loads the existing trained models from the `models/` folder instead. This makes rerunning the notebook much faster.

The trained models are not uploaded to GitHub. Therefore, anyone reproducing the task has to train the models once locally. After that, `skip_training = True` can be used to reuse the saved models from the `models/` folder.

The dataset is only downloaded once and reused afterwards.

## Data Setup

When running the notebook for this task, the data is downloaded automatically.

If the data already exists in:

```text
data/task_02/
```

the download step is skipped.

Otherwise, the dataset is downloaded and stored there.

After a successful download, the directory should contain the required spectra and label files for the task.

## Building the Report

After the notebook has been executed once, build the report from the repository root:

```bash
make TASK=task_02
```

This builds the report automatically using the generated figures and outputs the final PDF.
