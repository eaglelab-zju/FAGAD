import math
import os
import sys

import dgl
import numpy as np
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard
from torch import nn

current_file_name = __file__
current_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(current_file_name))))
if current_dir not in sys.path:
    sys.path.append(current_dir)


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()),
                         1 - nu,
                         axis=0)
    # if radius<0.1:
    #     radius=0.1

    return radius


def get_activation(name):
    activation_functions = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "tanh": F.tanh,
        "softmax": F.softmax,
        "elu": F.elu,
        "leaky_relu": F.leaky_relu,
        "prelu": F.prelu,
        "selu": F.selu,
        "gelu": F.gelu,
        "softplus": F.softplus,
        "softsign": F.softsign,
    }

    return activation_functions.get(name, None)


def bce_loss(preds, labels, norm=1.0, pos_weight=None):
    return norm * F.binary_cross_entropy_with_logits(
        preds,
        labels,
        pos_weight=pos_weight,
    )


def target_distribution(q):
    """get target distribution P

    Args:
        q (torch.Tensor): Soft assignments

    Returns:
        torch.Tensor: target distribution P
    """
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def init_weights(module: nn.Module, init="random") -> None:
    if isinstance(module, nn.Linear):
        # print('model init using::::::',init)
        if init == "random":
            nn.init.xavier_uniform_(module.weight.data)
        elif init == "zero":
            module.weight.data = ZerO_Init_on_matrix(module.weight.data)
        elif init == "identity":
            module.weight.data = Identity_Init_on_matrix(module.weight.data)

        if module.bias is not None:
            module.bias.data.fill_(0.0)


def Identity_Init_on_matrix(matrix_tensor):
    # Definition 1 in the paper
    # See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.eye_ for details. Preserves the identity of the inputs in Linear layers, where as many inputs are preserved as possible, the same as partial identity matrix.

    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)

    init_matrix = torch.nn.init.eye_(torch.empty(m, n))

    return init_matrix


def ZerO_Init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper. https://arxiv.org/abs/2110.12661

    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)

    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2**(clog_m)
        init_matrix = (torch.nn.init.eye_(torch.empty(m, p)) @ (
            torch.tensor(hadamard(p)).float() /
            (2**(clog_m / 2))) @ torch.nn.init.eye_(torch.empty(p, n)))

    return init_matrix
