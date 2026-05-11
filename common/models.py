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


class SpectraCNNFlowEncoder(nn.Module):
    """
    Configurable 1D convolutional encoder for conditional normalizing-flow models.

    Parameters
    ----------
    latent_dimension : int
        Number of output parameters required by the normalizing flow.
    channels : tuple[int, ...], optional
        Number of convolutional channels in each CNN block. Each entry creates one
        convolutional block.
    kernel_size : int, optional
        Kernel size used for all 1D convolutional layers.
    convs_per_block : int, optional
        Number of convolutional layers inside each block before pooling.
    pool_size : int, optional
        Kernel size of the max-pooling layer used at the end of each block.
    adaptive_pool_length : int, optional
        Fixed sequence length after adaptive average pooling. This makes the fully
        connected head independent of the original spectrum length.
    hidden_dims : tuple[int, ...], optional
        Sizes of the hidden fully connected layers after the convolutional feature
        extractor.
    dropout : float, optional
        Dropout probability applied after each hidden fully connected layer. Set to
        0.0 to disable dropout.
    use_batchnorm : bool, optional
        If True, applies batch normalization after each convolutional layer.
    activation : type[nn.Module], optional
        Activation function class used after convolutional and hidden linear layers.

    Input Shape
    -----------
    (batch_size, 1, spectrum_length)

    Output Shape
    ------------
    (batch_size, latent_dimension)

    Notes
    -----
    This encoder does not directly predict the stellar labels. Instead, it predicts
    the parameters of a conditional probability density. Label predictions and
    uncertainties are obtained by evaluating or sampling from the normalizing flow.
    """

    def __init__(
        self,
        latent_dimension: int,
        channels=(16, 32, 64),
        kernel_size: int = 9,
        convs_per_block: int = 1,
        pool_size: int = 4,
        adaptive_pool_length: int = 64,
        hidden_dims=(128,),
        dropout: float = 0.0,
        use_batchnorm: bool = True,
        activation=nn.ReLU,
    ):
        super().__init__()

        self.latent_dimension = latent_dimension

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
                    activation=activation,
                )
            )
            in_channels = out_channels

        self.features = nn.Sequential(
            *blocks,
            nn.AdaptiveAvgPool1d(adaptive_pool_length),
            nn.Flatten(),
        )

        feature_dim = channels[-1] * adaptive_pool_length

        head_layers = []
        previous_dim = feature_dim

        for hidden_dim in hidden_dims:
            head_layers.append(nn.Linear(previous_dim, hidden_dim))
            head_layers.append(activation())

            if dropout > 0:
                head_layers.append(nn.Dropout(dropout))

            previous_dim = hidden_dim

        head_layers.append(nn.Linear(previous_dim, latent_dimension))

        self.head = nn.Sequential(*head_layers)

    def _conv_block(
        self,
        in_channels,
        out_channels,
        kernel_size,
        convs_per_block,
        pool_size,
        use_batchnorm,
        activation,
    ):
        layers = []
        padding = kernel_size // 2

        for i in range(convs_per_block):
            current_in_channels = in_channels if i == 0 else out_channels

            layers.append(
                nn.Conv1d(
                    in_channels=current_in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                )
            )

            if use_batchnorm:
                layers.append(nn.BatchNorm1d(out_channels))

            layers.append(activation())

        layers.append(nn.MaxPool1d(pool_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x
