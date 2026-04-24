from common.setup_plotting import setup_matplotlib, get_figure_dir

import numpy as np


def main():
    setup_matplotlib()

    import matplotlib.pyplot as plt

    fig_dir = get_figure_dir(__file__)

    x = np.linspace(0, 10, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label=r"$\sin(x)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()
    fig.savefig(fig_dir / "sin_plot.pgf")
    plt.close(fig)


if __name__ == "__main__":
    main()
