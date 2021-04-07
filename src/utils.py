import itertools

import torch
import torch.nn as nn
import tensorly
tensorly.set_backend('pytorch')
from tensorly.tenalg import mode_dot


def update_configs(configs, default_configs) -> dict:
    if configs is None:
        return default_configs
    else:
        default_configs.update(configs)


def mse_loss(y, y_pred, indicators):
    mse = nn.MSELoss(reduction='sum')
    n_points = torch.sum(indicators)
    loss = mse(y*indicators, y_pred*indicators)
    if n_points < 1:
        return loss
    loss = loss / n_points
    return loss


def orthogonal_loss(param):
    with torch.enable_grad():
        sym = torch.mm(param, torch.t(param))
        eye = torch.eye(param.shape[0]).to(param.device)
        sym -= eye
        return sym.abs().sum()


def findsubsets(s, n):
    return list(itertools.combinations(s, n))


def findsubsets_all(s):
    n = len(s)
    subsets = []
    for i in range(1, n+1):
        subsets.extend(list(itertools.combinations(s, i)))
    return subsets


def mode_product(a: torch.tensor, b: torch.tensor, axis: int):
    tmp = b.transpose(0,1)
    return mode_dot(a, tmp, axis)
