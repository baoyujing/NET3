import torch
from torch import nn

from src import utils


class Output(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.configs = self.default_configs()
        utils.update_configs(configs=configs, default_configs=self.configs)

        self.n_outs = 1
        self.out_shape = []
        self.layer_list = nn.ModuleList()
        if self.configs["multiple"]:
            self._mlp_init()
        else:
            self._mlp_init_single()

    def forward(self, inputs):
        """
        :param inputs: [batch_size, n_sensor, n_type, dim_input]
        """
        if self.configs["multiple"]:
            return self._mlp_forward(inputs)
        else:
            output = self.layer_list[0](inputs)
            return output

    def _mlp_init(self):
        """
        Each time series has its own MLP.
        """
        for m, n in self.configs["mode_dims"].items():
            self.n_outs *= n
            self.out_shape.append(n)
        self.out_shape = tuple(self.out_shape)
        for n in range(self.n_outs):
            self.layer_list.append(nn.Linear(self.configs["dim_input"], 1))

    def _mlp_init_single(self):
        """
        Single MLP for all time series.
        """
        for m, n in self.configs["mode_dims"].items():
            self.out_shape.append(n)
        self.out_shape = tuple(self.out_shape)
        self.layer_list.append(nn.Linear(self.configs["dim_input"], 1))

    def _mlp_forward(self, inputs):
        """
        :param inputs: [batch_size, n_sensor, n_type, dim_input]
        """
        batch_size = inputs.size(0)
        inputs = inputs.reshape([batch_size, -1, self.configs["dim_input"]])

        outputs = []
        for n in range(self.n_outs):
            outputs.append(self.layer_list[n](inputs[:, n, :]))

        output_shape = list(self.out_shape)
        output_shape.insert(0, batch_size)
        return torch.stack(outputs, dim=1).reshape(output_shape)    # [n_sensor, n_type, 1]

    @classmethod
    def default_configs(cls):
        return {
            "dim_input": 8,
            "mode_dims": {0: 54, 1: 4},
            "multiple": True   # different outputs for different time series or single output for all time series
        }
