# common/models.py

import torch
from torch import nn


class SpectraCNN(nn.Module):
    def __init__(self, input_length, n_labels=3):
        super().__init__()

        out_channels = [16, 32, 64]
        kernel_sizes = [9, 9, 9]
        paddings = [4, 4, 4]

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=1, out_channels=out_channels[0], kernel_size=kernel_sizes[0], padding=paddings[0]),
            nn.ReLU(),
            # has stride equal to kernel size by default so reduces length by factor of 4
            nn.MaxPool1d(kernel_size=4),

            nn.Conv1d(out_channels[0], out_channels[1],
                      kernel_size=kernel_sizes[1], padding=paddings[1]),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(out_channels[1], out_channels[2],
                      kernel_size=kernel_sizes[2], padding=paddings[2]),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )

        # after 3 pooling layers with kernel size 4
        final_length = input_length // (4**3)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(out_channels[2] * final_length, 128),
            nn.ReLU(),
            nn.Linear(128, n_labels),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.head(x)
        return x


class SpectraUncertaintyCNN(nn.Module):
    """Configurable 1D CNN predicting means and log-variances."""

    def __init__(
        self,
        n_labels: int = 3,
        channels=(16, 32, 64),
        kernel_size: int = 3,
        convs_per_block: int = 2,
        pool_size: int = 4,
        adaptive_pool_length: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        use_batchnorm: bool = True,
        log_var_min: float = -10.0,
        log_var_max: float = 10.0,
    ):
        super().__init__()

        self.n_labels = n_labels
        self.log_var_min = log_var_min
        self.log_var_max = log_var_max

        blocks = []
        in_channels = 1

        for out_channels in channels:
            blocks.append(
                self._conv_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    convs_per_block=convs_per_block,
                    pool_size=pool_size,
                    use_batchnorm=use_batchnorm,
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool1d(adaptive_pool_length),
            nn.Flatten(),
        )

        feature_dim = channels[-1] * adaptive_pool_length

        self.shared_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.mu_head = nn.Linear(hidden_dim, n_labels)
        self.log_var_head = nn.Linear(hidden_dim, n_labels)

        nn.init.zeros_(self.log_var_head.weight)
        nn.init.zeros_(self.log_var_head.bias)

    def _conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        convs_per_block,
        pool_size,
        use_batchnorm,
    ):
        layers = []

        padding = kernel_size // 2

        for i in range(convs_per_block):
            current_in = in_channels if i == 0 else out_channels

            layers.append(
                nn.Conv1d(
                    current_in,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))

            layers.append(nn.ReLU())

        layers.append(nn.MaxPool1d(pool_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.features(x)
        hidden = self.shared_head(features)

        mu = self.mu_head(hidden)
        log_var = self.log_var_head(hidden)

        log_var = torch.clamp(
            log_var,
            min=self.log_var_min,
            max=self.log_var_max,
        )

        return mu, log_var


def gaussian_nll_loss(mu, log_var, y):
    """
    Gaussian NLL loss for predicted means and log variances.
    Same as in Lecture 5, but with log_var instead of sigma for better numerical stability. 

    log_var = log(sigma^2), so:
        NLL = 0.5 * (log_var + (y - mu)^2 / exp(log_var))

    Shapes:
        mu, log_var, y: (batch, 3)

    Returns:
        scalar mean loss.
    """
    loss = 0.5 * (log_var + (y - mu) ** 2 / torch.exp(log_var))
    return loss.mean()
