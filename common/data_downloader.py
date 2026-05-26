from pathlib import Path

from huggingface_hub import hf_hub_download


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# -----------------------------
# Task-specific download logic
# -----------------------------
def _task_01():
    dataset_dir = DATA_DIR / "task_01"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "labels.npy",
        "spectra.npy",
    ]

    # check if already downloaded
    if all((dataset_dir / f).exists() for f in required_files):
        print("task_01 data already available.")
        return dataset_dir

    print("Downloading data for task_01...")

    for file in required_files:
        hf_hub_download(
            repo_id="simbaswe/galah4",
            filename=file,
            repo_type="dataset",
            local_dir=dataset_dir,
        )

    print("Download complete.")
    return dataset_dir


def _task_04():
    dataset_dir = DATA_DIR / "task_04"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "train.pq",
        "val.pq",
        "test.pq",
    ]

    # Check if data is already available
    if all((dataset_dir / f).exists() for f in required_files):
        print("task_04 data already available.")
        return dataset_dir

    download_link = (
        "https://moodle.tu-dortmund.de/pluginfile.php/3652606/"
        "mod_folder/content/0/IceCube%202D%20Dataset.zip"
    )

    missing_files = [f for f in required_files if not (
        dataset_dir / f).exists()]

    raise FileNotFoundError(
        "task_04 data is missing.\n"
        f"Expected the following files in: {dataset_dir}\n"
        f"Missing files: {missing_files}\n\n"
        "Please download the dataset manually while being logged in to Moodle:\n"
        f"{download_link}\n\n"
        "After downloading, unzip the folder and place the files "
        "train.pq, val.pq, and test.pq in the task_04 data directory."
    )


# -----------------------------
# Task registry
# -----------------------------
TASKS = {
    "task_01": _task_01,
    # Tasks 1 through 3 used the same dataset.
    # "task_02": _task_02,
    # "task_03": _task_03,
    "task_04": _task_04,
    # add more tasks here
}


# -----------------------------
# Public interface
# -----------------------------
def download_data(task_name: str):
    if task_name not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")

    return TASKS[task_name]()
