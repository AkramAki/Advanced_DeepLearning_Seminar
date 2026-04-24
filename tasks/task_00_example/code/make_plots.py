from common.setup_plotting import setup_matplotlib, get_figure_dir

import numpy as np


def main():
    # Setup matplotlib BEFORE importing pyplot
    setup_matplotlib()

    import matplotlib.pyplot as plt

    fig_dir = get_figure_dir(__file__)

    # Example 1: simple function
    x = np.linspace(0, 10, 200)
    y = np.sin(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label=r"$\sin(x)$")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.legend()
    fig.savefig(fig_dir / "sine_plot.pdf")
    plt.close(fig)

    # Example 2: random "training curve"
    epochs = np.arange(1, 21)
    loss = np.exp(-epochs / 10) + 0.05 * np.random.rand(len(epochs))

    fig, ax = plt.subplots()
    ax.plot(epochs, loss, marker="o")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Example Training Curve")
    fig.savefig(fig_dir / "training_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
