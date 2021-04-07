import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from src.networks.tgcn_helper import TGCNHelper
from src import utils


class TGCN(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.configs = self.default_configs()
        utils.update_configs(configs, self.configs)

        self._idx2mode_sets()
        self._get_tgcn_dict()

        self.weight = Parameter(torch.FloatTensor(self.configs["dim_input"], self.configs["dim_output"]))
        self.bias = Parameter(torch.FloatTensor(self.configs["dim_output"]))

        self._reset_parameters()

    def forward(self, inputs, adj):
        """
        :param inputs: [batch_size, n_1, n_2, ..., dim]
        :param adj: {mode_id: adj matrix}
        :return h [batch_size, n_1, n_2, ..., dim]
        """
        h_list = [utils.mode_product(inputs, self.weight, axis=-1) + self.bias]
        for i, modes in self.idx2modes.items():
            adj_valid = self._get_valid_modes(adj, modes)
            layer = self.tgcn_dict[i]
            h = layer(inputs, adj_valid)
            h_list.append(h)

        h = torch.stack(h_list, dim=-1)
        h = torch.sum(h, dim=-1)
        h = F.relu(h)
        return h

    def _reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def _idx2mode_sets(self):
        mode_sets = utils.findsubsets_all(self.configs["valid_modes"])
        self.idx2modes = {str(i): modes for i, modes in enumerate(mode_sets)}

    def _get_helper(self, in_dims, out_dims, n_layers):
        helper = nn.ModuleList([TGCNHelper(out_dims, out_dims) for i in range(n_layers - 1)])
        helper.insert(0, TGCNHelper(in_dims, out_dims))
        return helper

    def _get_tgcn_dict(self):
        self.tgcn_dict = nn.ModuleDict()
        for i in self.idx2modes.keys():
            self.tgcn_dict[i] = \
                TGCNHelper(self.configs["dim_input"], self.configs["dim_output"])

    def _get_valid_modes(self, adj, modes):
        return {i: adj[i] for i in modes}

    @classmethod
    def default_configs(cls):
        return {
            'dim_input': 1,
            'dim_output': 8,
            "valid_modes": [0],  # [0]: valid mode is 0
        }
