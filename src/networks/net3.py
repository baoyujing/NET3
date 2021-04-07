import torch
from torch import nn

from src.networks.tgcn import TGCN
from src.networks.tlstm import TLSTM
from src.networks.output import Output
from src import utils


class NET3(nn.Module):
    def __init__(self, configs=None):
        super().__init__()
        self.configs = self.default_configs()
        utils.update_configs(configs, self.configs)

        self.tgcn = TGCN(self.configs["TGCN"])
        self.configs["TLSTM"]["dim_input"] = self.tgcn.configs["dim_output"]
        utils.update_configs(self.tgcn.configs, self.configs["TGCN"])

        self.configs["TLSTM"]["mode_dims"] = self.configs["mode_dims"]
        self.tlstm = TLSTM(self.configs["TLSTM"])
        utils.update_configs(self.tlstm.configs, self.configs["TLSTM"])

        self.configs["Output"]["mode_dims"] = self.configs["mode_dims"]
        self.configs["Output"]["dim_input"] = self.tgcn.configs["dim_output"] + self.tlstm.configs["dim_output"]
        self.output = Output(self.configs["Output"])

    def forward(self, values, adj, hx=None, indicators=None):
        """
        :param values: [batch_size, n_1, n_2, ..., n_M, t_step] values of tensors
        :param adj: adjacency matrix dictionary {n_m: A_m}
        :param hx: (h, c) [batch_size, n_1, n_2, ..., n_M, dim_emb] for both h and c
        :param indicators: [batch_size, n_1, n_2, ..., n_M, t_step] 0: missing value, 1: valid value
        """
        if indicators is None:
            indicators = torch.ones_like(values, dtype=torch.float)

        n_steps = values.shape[-1]
        out_list = []
        for t in range(n_steps):
            if t == 0:
                emb = (values[..., t]*indicators[..., t]).unsqueeze(-1)   # [batch_size, n_1, n_2, ..., n_M, dim]
            else:
                v = indicators[..., t]*values[..., t] + (1 - indicators[..., t])*out_list[-1]  # fill the missing values
                emb = (v * torch.ones_like(v)).unsqueeze(-1)

            emb_gcn = self.tgcn(inputs=emb, adj=adj)
            h_t, hx = self.tlstm(emb_gcn, hx)
            h_t = torch.cat([h_t, emb_gcn], dim=-1)
            out_t = self.output(h_t).squeeze()
            out_list.append(out_t)
        output = torch.stack(out_list, dim=-1)
        return output, hx

    @classmethod
    def default_configs(cls):
        return {
            "mode_dims": {0: 54, 1: 4},    # required for building the model
            "TGCN": TGCN.default_configs(),
            "TLSTM": TLSTM.default_configs(),
            "Output": Output.default_configs(),
        }
