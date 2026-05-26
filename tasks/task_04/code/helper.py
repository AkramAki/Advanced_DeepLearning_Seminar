import torch
import torch.nn as nn
from torch_geometric.nn import DynamicEdgeConv, global_mean_pool
from torch_geometric.data import Data, Batch

import random
import numpy as np

seed = 42

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def collate_fn_gnn(batch):
    """
    Custom function that defines how batches are formed.

    For a more complicated dataset with variable length per event and Graph Neural Networks,
    we need to define a custom collate function which is passed to the DataLoader.
    The default collate function in PyTorch Geometric is not suitable for this case.

    This function takes the Awkward arrays, converts them to PyTorch tensors,
    and then creates a PyTorch Geometric Data object for each event in the batch.

    You do not need to change this function.

    Parameters
    ----------
    batch : list
        A list of dictionaries containing the data and labels for each graph.
        The data is available in the "data" key and the labels are in the "xpos" and "ypos" keys.
    Returns
    -------
    packed_data : Batch
        A batch of graph data objects.
    labels : torch.Tensor
        A tensor containing the labels for each graph.
    """
    data_list = []
    labels = []

    for b in batch:
        # this is a loop over each event within the batch
        # b["data"] is the first entry in the batch with dimensions (n_features, n_hits)
        # where the features are (time, x, y)
        # for training a GNN, we need the graph notes, i.e., the individual hits, as the first dimension,
        # so we need to transpose to get (n_hits, n_features)
        tensor_data = torch.from_numpy(b["data"].to_numpy()).T
        # the original data is in double precision (float64), for our case single precision is sufficient
        # we let's convert to single precision (float32) to save memory and computation time
        tensor_data = tensor_data.to(dtype=torch.float32)

        # PyTorch Geometric needs the data in a specific format
        # we need to create a PyTorch Geometric Data object for each event
        this_graph_item = Data(x=tensor_data)
        data_list.append(this_graph_item)

        # also the labels need to be packaged as pytorch tensors
        labels.append(torch.Tensor([b["xpos"], b["ypos"]]).unsqueeze(0))

    # convert the list of tensors to a single tensor
    labels = torch.cat(labels, dim=0)
    # convert the list of Data objects to a single Batch object
    packed_data = Batch.from_data_list(data_list)
    return packed_data, labels


def MLP(channels):
    """
    Build a simple Multi-Layer Perceptron (MLP).

    Parameters
    ----------
    channels : list of int
        List containing the input dimension, hidden dimensions, and output dimension.
        For example, [6, 64, 64] creates:
        Linear(6 -> 64), ReLU, Linear(64 -> 64).

    Returns
    -------
    nn.Sequential
        A PyTorch sequential model containing linear layers and ReLU activations.
        ReLU is added after every linear layer except the final one.
    """
    layers = []

    for i in range(len(channels) - 1):
        layers.append(nn.Linear(channels[i], channels[i + 1]))

        if i < len(channels) - 2:
            layers.append(nn.ReLU())

    return nn.Sequential(*layers)


class GNNEncoder(nn.Module):
    def __init__(self, k=8):
        super(GNNEncoder, self).__init__()

        self.conv1 = DynamicEdgeConv(
            MLP([2 * 3, 64, 64]),
            k=k,
            aggr="mean"
        )

        self.conv2 = DynamicEdgeConv(
            MLP([2 * 64, 128, 128]),
            k=k,
            aggr="mean"
        )

        self.final_mlp = MLP([128, 64, 2])

    def forward(self, data):
        x = data.x
        batch = data.batch

        x = self.conv1(x, batch)
        x = self.conv2(x, batch)

        x = global_mean_pool(x, batch)

        x = self.final_mlp(x)

        return x
