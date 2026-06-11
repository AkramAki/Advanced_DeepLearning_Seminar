# Task 06

This task should run out of the box from the notebook in `code/task_06.ipynb`.
The report PDF can be rebuilt with:

```bash
make TASK=task_06
```

To fully recreate `report/main.pdf`, the notebook has to be run twice because the LaTeX report compares two saved figure sets:

1. Run the notebook with `left_mode_probability = 0.5` for the balanced `50/50` mixture.
   Rename or copy the generated figure files used in the report so they end in `_50`, for example `final_generated_vs_target_50.pdf`.
2. Run the notebook again with `left_mode_probability = 0.25`, which corresponds to `25%` left mode and `75%` right mode.
   Keep these generated files without the `_50` suffix, since these are the filenames used for the imbalanced case in `report/main.tex`.

This manual two-run workflow is not optimal, but it is sufficient for this submission.
