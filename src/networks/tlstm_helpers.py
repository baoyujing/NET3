import math
import torch
import torch.nn as nn

from src.utils import mode_product, orthogonal_loss


class TensorLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, in_modes_dict, out_modes_dicts):
        super().__init__()
        self.w = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.b = nn.Parameter(torch.FloatTensor(out_features))

        self.w_dict = nn.ParameterDict(
            {str(k): nn.Parameter(torch.FloatTensor(in_modes_dict[k], out_modes_dicts[k])) for k in in_modes_dict})

        self.reset_parameters()

    def forward(self, x):
        """
        :param x: [batch_size, n_1, n_2, ,,, n_m, in_features]
        """
        h_x = mode_product(x, self.w, axis=-1)
        for k in self.w_dict:
            h_x = mode_product(h_x, self.w_dict[k], axis=int(k)+1)  # first dim is batch
        return h_x + self.b

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
        self.b.data.copy_(torch.zeros(self.w.size(1)))

        for k, w in self.w_dict.items():
            stdv = 1./math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)


class TuckerLayer(nn.Module):
    def __init__(self, in_modes_dict, out_modes_dicts):
        super().__init__()
        self.u_dict = nn.ParameterDict(
            {str(k): nn.Parameter(torch.FloatTensor(in_modes_dict[k], out_modes_dicts[k])) for k in in_modes_dict})

        self._reset_parameters()

    def forward(self, x, back=False):
        """
        :param x: [batch_size, n_1, n_2, ,,, n_m, in_features]
        :back: False: Decomposition, True: reconstruction
        """
        if back:
            return self._backward(x)
        return self._forward(x)

    def _forward(self, x):
        """
        Decomposition
        """
        for k in self.u_dict:
            x = mode_product(x, self.u_dict[k], axis=int(k)+1)
        return x

    def _backward(self, x):
        """
        Reconstruction.
        """
        for k in self.u_dict:
            x = mode_product(x, self.u_dict[k].permute([1, 0]), axis=int(k)+1)
        return x

    def _reset_parameters(self):
        for k, w in self.u_dict.items():
            stdv = 1. / math.sqrt(w.size(1))
            w.data.uniform_(-stdv, stdv)

    def get_orthogonal_loss(self):
        loss = 0.0
        for k, w in self.u_dict.items():
            loss += orthogonal_loss(w)
        return loss
