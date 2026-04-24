from common.setup_plotting import setup_matplotlib, get_figure_dir
from common.data_downloader import download_data

import numpy as np


def main():
    setup_matplotlib()        # configure matplotlib first
    import matplotlib.pyplot as plt   # THEN import pyplot

    download_data("task_01")  # download data for this task, if needed

    fig_dir = get_figure_dir(__file__)

    x = np.linspace(0, 10, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label=r"$\sin(x)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()
    fig.savefig(fig_dir / "sin_plot.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
