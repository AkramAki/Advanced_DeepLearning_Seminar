## Data Setup

When running `make` for this task, the data is downloaded automatically.

- If the data already exists in  
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
