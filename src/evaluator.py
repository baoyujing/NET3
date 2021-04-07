import torch
import logging
from tqdm import tqdm

from src.networks.net3 import NET3
from src.data_iterator import DataIterator

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, configs, model: NET3, iterator: DataIterator, task="missing"):
        self.configs = self.default_configs()
        utils.update_configs(configs, self.configs)

        self.task = task
        self.model = model

        if self.configs["pretrained_path"] is not None:
            self.model.load_state_dict(torch.load(self.configs["pretrained_path"]))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.iterator = iterator
        self.networks = self.iterator.get_networks()
        for n in self.networks:
            self.networks[n] = self.networks[n].to(self.device)

        self.loss_calc = utils.mse_loss

    def eval(self):
        if self.task == "missing":
            return self.eval_missing()
        if self.task == "future":
            return self.eval_future()

    def eval_missing(self):
        with torch.no_grad():
            epoch_loss = 0
            n_points = 0
            for values, indicators_train, indicators_eval in tqdm(self.iterator):
                values = values.to(self.device)
                indicators_train = indicators_train.to(self.device)

                y, y_pred = self._iter(values=values, indicators=indicators_train)
                indicators_y = indicators_eval[..., -1].to(self.device)

                n_point = torch.sum(indicators_y)
                n_points += n_point
                epoch_loss += self.get_loss(y=y, y_pred=y_pred, indicators=indicators_y) * n_point
            epoch_loss = torch.sqrt(epoch_loss/n_points)
            logger.info("Evaluation loss: {}, n points: {}".format(epoch_loss, n_points))
            return epoch_loss

    def eval_future(self):
        with torch.no_grad():
            epoch_loss = 0
            n_points = 0
            for values, indicators_train, indicators_eval in tqdm(self.iterator):  # [n_sensor, n_type, n_step]
                values = values.to(self.device)

                # skip training data
                if torch.sum(indicators_eval) < 1:
                    continue

                # the last time step is used for evaluation
                indicators_train = indicators_train.to(self.device)
                indicators_eval = indicators_eval.to(self.device)
                indicators_train += indicators_eval
                indicators_train[..., -1] = 0
                indicators_eval[..., :-1] = 0

                y, y_pred = self._iter(values=values, indicators=indicators_train)
                indicators_y = indicators_eval[..., -1].to(self.device)

                n_point = torch.sum(indicators_y)
                n_points += n_point
                epoch_loss += self.get_loss(y=y, y_pred=y_pred, indicators=indicators_y) * n_point
            epoch_loss = torch.sqrt(epoch_loss/n_points)
            logger.info("Evaluation loss: {}.".format(epoch_loss))
            return epoch_loss

    def _iter(self, values, indicators):
        y_pred, hx = self.model(values=values[..., :-1], indicators=indicators[..., :-1], adj=self.networks)
        return values[..., -1], y_pred[..., -1]

    def get_loss(self, y, y_pred, indicators):
        return self.loss_calc(y=y, y_pred=y_pred, indicators=indicators)

    @classmethod
    def default_configs(cls):
        return {
            "pretrained_path": None,
        }
