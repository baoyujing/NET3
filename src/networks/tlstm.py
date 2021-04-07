import torch
import torch.nn as nn

from src.networks.tlstm_helpers import TensorLinearLayer, TuckerLayer
from src import utils


class TLSTM(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.configs = self.default_configs()
        utils.update_configs(configs, self.configs)

        if self.configs["is_decompose"]:
            self.tucker_layer = TuckerLayer(
                in_modes_dict=self.configs["mode_dims"], out_modes_dicts=self.configs["mode_dims_hidden"])
        else:
            self.configs["mode_dims_hidden"] = self.configs["mode_dims"]

        self.l_xi = TensorLinearLayer(
            in_features=self.configs["dim_input"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_hi = TensorLinearLayer(
            in_features=self.configs["dim_output"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_xf = TensorLinearLayer(
            in_features=self.configs["dim_input"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_hf = TensorLinearLayer(
            in_features=self.configs["dim_output"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_xg = TensorLinearLayer(
            in_features=self.configs["dim_input"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_hg = TensorLinearLayer(
            in_features=self.configs["dim_output"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_xo = TensorLinearLayer(
            in_features=self.configs["dim_input"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])
        self.l_ho = TensorLinearLayer(
            in_features=self.configs["dim_output"], out_features=self.configs["dim_output"],
            in_modes_dict=self.configs["mode_dims_hidden"], out_modes_dicts=self.configs["mode_dims_hidden"])

        self.out_shape = [dim for i, dim in self.configs["mode_dims_hidden"].items()]
        self.out_shape.append(self.configs["dim_output"])

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.reconstruction_loss = 0.0
        self.criterion = nn.MSELoss()

    def forward(self, x, hx=None):
        """
        :param x: [batch_size, n_1, n_2, ..., n_m, in_features]
        :param hx: (h ,c)  [batch_size, out_features]
        :return:
        """
        h, c = self._init_parameters(x=x, hx=hx)

        if self.configs["is_decompose"]:
            z = self.tucker_layer(x)
            self.reconstruction_loss = torch.sqrt(self.criterion(x, self.tucker_layer(z, True)))
        else:
            z = x

        f = self.sigmoid(self.l_xf(z) + self.l_hf(h))
        i = self.sigmoid(self.l_xi(z) + self.l_hi(h))
        o = self.sigmoid(self.l_xo(z) + self.l_ho(h))
        g = self.tanh(self.l_xg(z) + self.l_hg(h))
        c = f * c + i * g
        h = o * self.tanh(c)

        hx = (h, c)

        if self.configs["is_decompose"]:
            h = self.tucker_layer(h, True)
        return h, hx

    def get_orthogonal_loss(self):
        return self.tucker_layer.get_orthogonal_loss()

    def get_reconstruction_loss(self):
        return self.reconstruction_loss

    def _init_parameters(self, x, hx=None):
        outshape = [x.size(0)]
        outshape.extend(self.out_shape)
        if hx is None:
            h = torch.zeros(outshape)
            c = torch.zeros(outshape)
        else:
            h, c = hx
        device = x.device
        return h.to(device), c.to(device)

    @classmethod
    def default_configs(cls):
        return {
            "dim_input": 8,
            "dim_output": 8,
            "mode_dims": {0: 54, 1: 4},
            "mode_dims_hidden": {0: 15, 1: 2},
            "is_decompose": True,
        }
