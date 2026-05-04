import torch


def gaussian_nll_loss(outputs, y):
    """
    Gaussian NLL loss for predicted means and log variances.

    outputs = (mu, log_var), where log_var = log(sigma^2).

    Ignoring the constant 0.5 * log(2*pi):
        NLL = 0.5 * (log_var + (y - mu)^2 / exp(log_var))

    Shapes:
        mu, log_var, y: (batch_size, n_targets)

    Returns:
        Mean scalar loss.
    """
    mu, log_var = outputs
    loss = 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))
    return loss.mean()
