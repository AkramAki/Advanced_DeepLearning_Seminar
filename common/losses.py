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


def nf_loss(inputs, batch_labels, model) -> torch.Tensor:
    """
    Computes the loss for a normalizing flow model.

    Parameters
    ----------
    inputs : torch.Tensor
        The input data to the model.
    batch_labels : torch.Tensor
        The labels corresponding to the input data.
    model : torch.nn.Module
        The normalizing flow model used for evaluation.
    Returns
    -------
    torch.Tensor
        The computed loss value.
    """
    log_pdfs = model.log_pdf_evaluation(
        batch_labels, inputs)  # get the probability of the labels given the input data
    loss = -log_pdfs.mean()  # take the negative mean of the log probabilities
    return loss
