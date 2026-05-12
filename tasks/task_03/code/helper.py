from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from IPython.display import display, Image

from sklearn.metrics import mean_absolute_error, r2_score


def denormalize(x, mean, std):
    """
    Convert normalized labels back to original scale.
    """
    return x * std + mean


def collect_model_predictions(
    model,
    test_loader,
    device,
    label_mean,
    label_std,
    n_labels,
    samplesize_model=1000,
    max_batches=None,
):
    """
    Run the trained model on the test set and return predictions,
    uncertainties, truths, and one example batch for PDF plots.
    """

    label_mean = np.asarray(label_mean)
    label_std = np.asarray(label_std)

    model.eval()

    all_pred_means = []
    all_pred_stds = []
    all_truths = []

    first_batch_x = None
    first_batch_y = None

    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if first_batch_x is None:
                first_batch_x = batch_x
                first_batch_y = batch_y

            preds = model(
                batch_x,
                samplesize_per_batchitem=samplesize_model,
            )

            pred_means = preds[:, :n_labels]
            pred_stds = preds[:, n_labels: 2 * n_labels]

            all_pred_means.append(pred_means.detach().cpu())
            all_pred_stds.append(pred_stds.detach().cpu())
            all_truths.append(batch_y.detach().cpu())

    pred_means_norm = torch.cat(all_pred_means, dim=0).numpy()
    pred_stds_norm = torch.cat(all_pred_stds, dim=0).numpy()
    truths_norm = torch.cat(all_truths, dim=0).numpy()

    predictions_orig = denormalize(pred_means_norm, label_mean, label_std)
    true_labels_orig = denormalize(truths_norm, label_mean, label_std)
    sigmas_orig = pred_stds_norm * label_std

    return {
        "predictions_orig": predictions_orig,
        "true_labels_orig": true_labels_orig,
        "sigmas_orig": sigmas_orig,
        "first_batch_x": first_batch_x,
        "first_batch_y": first_batch_y,
    }


def plot_pdf_predictions(
    model,
    input_data,
    truth,
    fig_dir,
    star_indices=(0, 1, 2, 3, 4),
    samplesize_pdf=5000,
    filename_template="pdf_prediction_star_{i}.png",
    show=True,
    image_width=700,
):
    """
    Save the model PDF prediction plots for selected stars.
    """

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []

    for i in star_indices:
        filename = fig_dir / filename_template.format(i=i)

        model.visualize_pdf(
            input_data=input_data,
            filename=str(filename),
            samplesize=samplesize_pdf,
            batch_index=i,
            truth=truth[i],
        )

        saved_paths.append(filename)

        if show:
            display(Image(filename=str(filename), width=image_width))

    return saved_paths


def plot_predictions_vs_true(
    true_labels_orig,
    predictions_orig,
    sigmas_orig,
    fig_dir,
    latex_names,
    filename="predictions_vs_true_with_uncertainty.pdf",
    figsize=(6, 11),
    point_size=4,
    point_alpha=0.5,
    show=True,
    close=False,
):
    """
    Plot true labels against predicted labels with uncertainty bars.
    The labels are stacked vertically instead of placed side-by-side.
    """

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_labels = len(latex_names)

    fig, axes = plt.subplots(
        n_labels,
        1,
        figsize=figsize,
        sharex=False,
    )

    if n_labels == 1:
        axes = [axes]

    metrics = {}

    for i, ax in enumerate(axes):
        x_true = true_labels_orig[:, i]
        y_pred = predictions_orig[:, i]
        y_err = sigmas_orig[:, i]

        ax.errorbar(
            x_true,
            y_pred,
            yerr=y_err,
            fmt="o",
            ms=point_size,
            alpha=point_alpha,
            color="C0",
            ecolor="C0",
            elinewidth=0.8,
            capsize=1.5,
            mec="k",
            mew=0.3,
            linestyle="none",
            label="Test Set",
        )

        lo = min(x_true.min(), y_pred.min())
        hi = max(x_true.max(), y_pred.max())
        pad = 0.03 * (hi - lo)

        ax.plot(
            [lo - pad, hi + pad],
            [lo - pad, hi + pad],
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="Ideal fit",
            zorder=3,
        )

        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        mae = mean_absolute_error(x_true, y_pred)
        r2 = r2_score(x_true, y_pred)

        metrics[latex_names[i]] = {
            "mae": mae,
            "r2": r2,
        }

        ax.set_xlabel(f"True {latex_names[i]}")
        ax.set_ylabel(f"Predicted {latex_names[i]}")
        ax.set_title(latex_names[i])

        metrics_text = f"MAE: {mae:.4f}\n$R^2$: {r2:.4f}"

        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.75,
                edgecolor="0.8",
            ),
        )

        ax.legend(loc="lower right", frameon=True)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Model Predictions vs True Labels on Test Set",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout()

    saved_path = fig_dir / filename
    fig.savefig(saved_path, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return saved_path, metrics


def plot_uncertainty_vs_error(
    true_labels_orig,
    predictions_orig,
    sigmas_orig,
    fig_dir,
    latex_names,
    filename="uncertainty_vs_absolute_error.pdf",
    figsize=(6, 11),
    point_size=12,
    point_alpha=0.55,
    show=True,
    close=False,
):
    """
    Plot predicted uncertainty against absolute prediction error.

    A well-calibrated model should generally show larger errors for
    objects with larger predicted uncertainty.
    """

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_labels = len(latex_names)

    residuals = predictions_orig - true_labels_orig
    abs_errors = np.abs(residuals)

    fig, axes = plt.subplots(
        n_labels,
        1,
        figsize=figsize,
        sharex=False,
    )

    if n_labels == 1:
        axes = [axes]

    metrics = {}

    for i, ax in enumerate(axes):
        sigma = sigmas_orig[:, i]
        abs_error = abs_errors[:, i]

        ax.scatter(
            sigma,
            abs_error,
            s=point_size,
            alpha=point_alpha,
            color="C0",
            edgecolor="k",
            linewidth=0.25,
            label="Test Set",
        )

        max_sigma = sigma.max()
        max_abs_error = abs_error.max()
        max_x = max_sigma
        max_y = max(max_abs_error, 2 * max_sigma)

        ax.plot(
            [0, max_x],
            [0, max_x],
            color="red",
            linestyle="--",
            linewidth=1.2,
            label=r"$|\Delta| = 1\sigma$",
        )

        ax.plot(
            [0, max_x],
            [0, 2 * max_x],
            color="C1",
            linestyle=":",
            linewidth=1.4,
            label=r"$|\Delta| = 2\sigma$",
        )

        ax.set_xlim(0, 1.05 * max_x)
        ax.set_ylim(0, 1.05 * max_y)

        median_sigma = np.median(sigma)
        median_abs_error = np.median(abs_error)
        mean_sigma = np.mean(sigma)
        mean_abs_error = np.mean(abs_error)
        frac_within_1sigma = np.mean(abs_error <= sigma)
        frac_within_2sigma = np.mean(abs_error <= 2 * sigma)

        metrics[latex_names[i]] = {
            "median_sigma": median_sigma,
            "median_absolute_error": median_abs_error,
            "mean_sigma": mean_sigma,
            "mean_absolute_error": mean_abs_error,
            "frac_within_1sigma": frac_within_1sigma,
            "frac_within_2sigma": frac_within_2sigma,
        }

        metrics_text = (
            f"Median $\\sigma$: {median_sigma:.4f}\n"
            f"Median $|\\Delta|$: {median_abs_error:.4f}\n"
            f"Within $1\\sigma$: {100 * frac_within_1sigma:.1f}\\%\n"
            f"Within $2\\sigma$: {100 * frac_within_2sigma:.1f}\\%"
        )
        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.75,
                edgecolor="0.8",
            ),
        )

        ax.set_xlabel(f"Predicted uncertainty in {latex_names[i]}")
        ax.set_ylabel(f"Absolute error in {latex_names[i]}")
        ax.set_title(f"Uncertainty vs Error: {latex_names[i]}")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Predicted Uncertainty vs Absolute Error",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout()

    saved_path = fig_dir / filename
    fig.savefig(saved_path, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return saved_path, metrics


def plot_residual_distributions(
    true_labels_orig,
    predictions_orig,
    fig_dir,
    latex_names,
    filename="residual_distributions.pdf",
    figsize=(6, 11),
    bins=30,
    show=True,
    close=False,
):
    """
    Plot residual distributions for all labels.

    Residual is defined as:

        prediction - truth
    """

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    n_labels = len(latex_names)
    residuals = predictions_orig - true_labels_orig

    fig, axes = plt.subplots(
        n_labels,
        1,
        figsize=figsize,
        sharex=False,
    )

    if n_labels == 1:
        axes = [axes]

    metrics = {}

    for i, ax in enumerate(axes):
        res = residuals[:, i]

        mean_res = np.mean(res)
        std_res = np.std(res)
        median_res = np.median(res)

        q16, q84 = np.percentile(res, [16, 84])

        metrics[latex_names[i]] = {
            "mean_residual": mean_res,
            "median_residual": median_res,
            "std_residual": std_res,
            "q16": q16,
            "q84": q84,
        }

        ax.hist(
            res,
            bins=bins,
            density=True,
            alpha=0.65,
            color="C0",
            edgecolor="black",
            linewidth=0.5,
        )

        ax.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=1.2,
            label="Zero residual",
        )

        ax.axvline(
            mean_res,
            color="C1",
            linestyle="-",
            linewidth=1.2,
            label="Mean residual",
        )

        ax.axvspan(
            q16,
            q84,
            color="C0",
            alpha=0.15,
            label="16th–84th percentile",
        )

        metrics_text = (
            f"Mean: {mean_res:.4f}\n"
            f"Median: {median_res:.4f}\n"
            f"Std: {std_res:.4f}"
        )

        ax.text(
            0.05,
            0.95,
            metrics_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            bbox=dict(
                boxstyle="round",
                facecolor="white",
                alpha=0.75,
                edgecolor="0.8",
            ),
        )

        ax.set_xlabel(f"Residual in {latex_names[i]}")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual Distribution: {latex_names[i]}")
        ax.legend(loc="upper right", frameon=True)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Residual Distributions",
        fontsize=15,
        y=0.995,
    )

    fig.tight_layout()

    saved_path = fig_dir / filename
    fig.savefig(saved_path, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return saved_path, metrics


def run_prediction_diagnostics(
    model,
    test_loader,
    device,
    fig_dir,
    label_mean,
    label_std,
    n_labels=3,
    latex_names=None,
    samplesize_model=1000,
    samplesize_pdf=5000,
    pdf_star_indices=(0, 1, 2, 3, 4),
    max_batches=None,
    make_pdf_predictions=True,
    make_predictions_vs_true=True,
    make_uncertainty_vs_error=True,
    make_residual_distributions=True,
    show=True,
    close=False,
    figsize=(6, 11),
):
    """
    Main convenience function.

    Generates:
    1. PDF prediction star PNGs
    2. stacked predictions-vs-true plot
    3. uncertainty-vs-absolute-error plot
    4. residual distribution plot
    """

    if latex_names is None:
        latex_names = [
            r"$T_{\mathrm{eff}}$",
            r"$\log g$",
            r"$[\mathrm{Fe}/\mathrm{H}]$",
        ]

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    prediction_data = collect_model_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
        label_mean=label_mean,
        label_std=label_std,
        n_labels=n_labels,
        samplesize_model=samplesize_model,
        max_batches=max_batches,
    )

    predictions_orig = prediction_data["predictions_orig"]
    true_labels_orig = prediction_data["true_labels_orig"]
    sigmas_orig = prediction_data["sigmas_orig"]

    saved_paths = {}
    metrics = {}

    if make_pdf_predictions:
        saved_paths["pdf_predictions"] = plot_pdf_predictions(
            model=model,
            input_data=prediction_data["first_batch_x"],
            truth=prediction_data["first_batch_y"],
            fig_dir=fig_dir,
            star_indices=pdf_star_indices,
            samplesize_pdf=samplesize_pdf,
            show=show,
        )

    if make_predictions_vs_true:
        path, plot_metrics = plot_predictions_vs_true(
            true_labels_orig=true_labels_orig,
            predictions_orig=predictions_orig,
            sigmas_orig=sigmas_orig,
            fig_dir=fig_dir,
            latex_names=latex_names,
            show=show,
            close=close,
            figsize=figsize,
        )

        saved_paths["predictions_vs_true"] = path
        metrics["predictions_vs_true"] = plot_metrics

    if make_uncertainty_vs_error:
        path, plot_metrics = plot_uncertainty_vs_error(
            true_labels_orig=true_labels_orig,
            predictions_orig=predictions_orig,
            sigmas_orig=sigmas_orig,
            fig_dir=fig_dir,
            latex_names=latex_names,
            show=show,
            close=close,
            figsize=figsize,
        )

        saved_paths["uncertainty_vs_error"] = path
        metrics["uncertainty_vs_error"] = plot_metrics

    if make_residual_distributions:
        path, plot_metrics = plot_residual_distributions(
            true_labels_orig=true_labels_orig,
            predictions_orig=predictions_orig,
            fig_dir=fig_dir,
            latex_names=latex_names,
            show=show,
            close=close,
            figsize=figsize,
        )

        saved_paths["residual_distributions"] = path
        metrics["residual_distributions"] = plot_metrics

    return {
        "predictions_orig": predictions_orig,
        "true_labels_orig": true_labels_orig,
        "sigmas_orig": sigmas_orig,
        "saved_paths": saved_paths,
        "metrics": metrics,
    }
