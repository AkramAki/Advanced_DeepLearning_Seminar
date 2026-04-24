from pathlib import Path

from huggingface_hub import hf_hub_download


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


# -----------------------------
# Task-specific download logic
# -----------------------------
def _task_01() -> None:
    dataset_dir = DATA_DIR / "task_01"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    required_files = [
        "labels.npy",
        "spectra.npy",
    ]

    # check if already downloaded
    if all((dataset_dir / f).exists() for f in required_files):
        print("task_01 data already available.")
        return

    print("Downloading data for task_01...")

    for file in required_files:
        hf_hub_download(
            repo_id="simbaswe/galah4",
            filename=file,
            repo_type="dataset",
            local_dir=dataset_dir,
        )

    print("Download complete.")


# -----------------------------
# Task registry
# -----------------------------
TASKS = {
    "task_01": _task_01,
    # "task_02": _task_02,
    # add more tasks here
}


# -----------------------------
# Public interface
# -----------------------------
def download_data(task_name: str) -> None:
    if task_name not in TASKS:
        available = ", ".join(TASKS)
        raise ValueError(f"Unknown task '{task_name}'. Available: {available}")

    TASKS[task_name]()
