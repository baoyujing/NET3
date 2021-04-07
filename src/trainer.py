import os
import logging
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.optim as optim

from src.evaluator import Evaluator
from src.data_iterator import DataIterator
from src.networks import NET3

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, configs, model: NET3, iterator: DataIterator, is_eval=False):
        # configs
        self.configs = self.default_configs()
        utils.update_configs(configs, self.configs)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = model
        if self.configs["pretrained_path"] is not None:
            self.model.load_state_dict(torch.load(self.configs["pretrained_path"]))

        # iterator
        self.iterator = iterator
        self.networks = self.iterator.get_networks()
        for n in self.networks:
            self.networks[n] = self.networks[n].to(self.device)

        # loss and optimizer
        self.loss_calc = utils.mse_loss
        self.opt = optim.Adam(params=self.model.parameters(), lr=self.configs["lr"])
        self.loss_min = np.inf
        self.n_tolerance = 0

        # path
        if not os.path.exists(self.configs["save_dir"]):
            os.makedirs(self.configs["save_dir"])

        # evaluation
        self.is_eval = is_eval
        if is_eval:
            configs = self.iterator.configs
            configs["mode"] = "eval"
            self.iterator_eval = DataIterator(configs=configs)

    def train(self):
        n_iter = 0
        for epoch in range(self.configs["n_epoch"]):
            logger.info("Epoch: {}".format(epoch))
            epoch_loss = 0
            for values, indicators_train, indicators_eval in tqdm(self.iterator):
                values = values.to(self.device)
                indicators_train = indicators_train.to(self.device)

                self.opt.zero_grad()
                y, y_pred, indicators_y = self._iter(values=values, indicators=indicators_train)
                iter_loss = self.get_loss(y=y, y_pred=y_pred, indicators=indicators_y)
                iter_loss.backward()
                self.opt.step()

                n_iter += 1
                epoch_loss += iter_loss
            logger.info("Training loss: {}".format(epoch_loss/len(self.iterator)))

            if self.is_eval:
                evaluator = Evaluator(configs=None, model=self.model, iterator=self.iterator_eval)
                eval_loss = evaluator.eval()
                logger.info("Eval loss: {}".format(eval_loss))
                if self.early_stopping(loss=eval_loss):
                    return
            else:
                if self.early_stopping(loss=epoch_loss/len(self.iterator)):
                    return

    def _iter(self, values, indicators):
        y_pred, hx = self.model(values=values[..., :-1], indicators=indicators[..., :-1], adj=self.networks)
        return values[..., -1], y_pred[..., -1], indicators[..., -1]

    def get_loss(self, y, y_pred, indicators):
        if self.model.configs["TLSTM"]["is_decompose"]:
            return self.get_loss_rmse(y, y_pred, indicators) + \
                   self.configs["orthogonal_weight"] * self.get_loss_orthogonal() + \
                   self.configs["reconstruct_weight"] * self.get_loss_reconstruction()
        return self.get_loss_rmse(y, y_pred, indicators)

    def get_loss_rmse(self, y, y_pred, indicators):
        return torch.sqrt(self.loss_calc(y=y, y_pred=y_pred, indicators=indicators))

    def get_loss_orthogonal(self):
        return self.model.tlstm.get_orthogonal_loss()

    def get_loss_reconstruction(self):
        return self.model.tlstm.get_reconstruction_loss()

    def early_stopping(self, loss):
        if loss < self.loss_min:
            self.loss_min = loss
            self.n_tolerance = 0
            torch.save(self.model.state_dict(), os.path.join(self.configs["save_dir"], "model.train.ckpt"))
        else:
            self.n_tolerance += 1
        if self.n_tolerance >= self.configs["tolerance"]:
            warnings.warn("Early stopped.")
            return True
        return False

    @classmethod
    def default_configs(cls):
        return {
            "n_epoch": 50,
            "lr": 0.01,
            "tolerance": 5,
            "orthogonal_weight": 1e-3,
            "reconstruct_weight": 1e-3,
            "pretrained_path": None
        }
