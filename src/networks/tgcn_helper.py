import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from src import utils


class TGCNHelper(Module):
    """
    Batch Tensor GCN layer.
    """
    def __init__(self, in_features, out_features):
        super(TGCNHelper, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs: torch.tensor, adj_dict):
        """
        :param inputs: [batch_size, n_1, n_2, ..., n_order, dim] tensor
        :param adj_dict:  {mode_id: network}
        """
        outputs = utils.mode_product(inputs, self.weight, axis=-1)
        for n in adj_dict:
            outputs = utils.mode_product(outputs, adj_dict[n], axis=n + 1)
        return outputs
