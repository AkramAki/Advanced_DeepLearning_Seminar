from pathlib import Path

import matplotlib as mpl


def setup_matplotlib():
    repo_dir = Path(__file__).resolve().parents[1]

    mpl.use("pgf")

    rc_path = repo_dir / "latex" / "matplotlibrc"
    header_path = repo_dir / "latex" / "header-matplotlib.tex"

    mpl.rc_file(rc_path)

    # Use absolute path so LaTeX finds the header from anywhere
    mpl.rcParams["pgf.preamble"] = rf"\input{{{header_path.as_posix()}}}"

    return repo_dir


def get_figure_dir(task_file: str) -> Path:
    task_dir = Path(task_file).resolve().parents[1]
    build_dir = task_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    return build_dir
