import os
import torch
import pickle
import logging
import numpy as np

from src import utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIterator:
    def __init__(self, configs=None):
        logging.log(level=logging.INFO, msg="Initialize DataIterator.")
        self.configs = self.default_configs()
        utils.update_configs(configs=configs, default_configs=self.configs)

        self.values = pickle.load(open(
            os.path.join(self.configs["data_root"], self.configs["values_name"]), "rb"))
        self.indicators_train = pickle.load(open(
            os.path.join(self.configs["data_root"], self.configs["indicators_train_name"]), "rb"))
        self.indicators_eval = pickle.load(open(
            os.path.join(self.configs["data_root"], self.configs["indicators_eval_name"]), "rb"))
        self.max_step = self.values.shape[-1]

        if self.configs["mode"] == "eval":
            self.configs["stride"] = 1
            self.configs["shuffle"] = False

        self.n_iter = int(np.ceil((self.max_step - self.configs["window_size"]) / self.configs["stride"])) + 1
        if self.configs["batch_size"] > 1:
            self.shapes = list(self.values.shape)[:-1]
            self._pad()

        self.starts = [i * self.configs["stride"] for i in range(self.n_iter)]
        self.ends = [i + self.configs["window_size"] for i in self.starts]
        self.start_ends = list(zip(self.starts, self.ends))

        # handle batch
        self.n_iter = int(np.ceil(self.n_iter / self.configs["batch_size"]))
        self.starts_batch = [i * self.configs["batch_size"] for i in range(self.n_iter)]
        self.ends_batch = [i + self.configs["batch_size"] for i in self.starts_batch]

    def __iter__(self):
        if self.configs["shuffle"]:
            np.random.shuffle(self.start_ends)

        for i in range(self.n_iter):
            start_batch = self.starts_batch[i]
            values_list = []
            indicators_train_list = []
            indicators_eval_list = []
            starts_ends = self.start_ends[start_batch: start_batch + self.configs["batch_size"]]
            for start, end in starts_ends:
                values_list.append(self.values[..., start: end])
                indicators_train_list.append(self.indicators_train[..., start: end])
                indicators_eval_list.append(self.indicators_eval[..., start: end])
            values = torch.from_numpy(np.stack(values_list, axis=0)).float()
            indicators_train = torch.from_numpy(np.stack(indicators_train_list, axis=0)).float()
            indicators_eval = torch.from_numpy(np.stack(indicators_eval_list, axis=0)).float()
            yield values, indicators_train, indicators_eval

    def _pad(self):
        max_step = self.n_iter * self.configs["stride"] + self.configs["window_size"]
        diff = max_step - self.max_step
        self.shapes.append(diff)
        pad_tensor = np.zeros(self.shapes)
        self.values = np.concatenate([self.values, pad_tensor], axis=-1)
        self.indicators_train = np.concatenate([self.indicators_train, pad_tensor], axis=-1)
        self.indicators_eval = np.concatenate([self.indicators_eval, pad_tensor], axis=-1)

    def get_networks(self):
        networks = pickle.load(open(
            os.path.join(self.configs["data_root"], self.configs["network_name"]), "rb"))
        for n in networks:
            networks[n] = torch.from_numpy(networks[n]).float()
        return networks

    def __len__(self):
        return self.n_iter

    @classmethod
    def default_configs(cls):
        return {
            "window_size": 6,   # 5 (historical) + 1 (future)
            "batch_size": 1,
            "stride": 1,
            "shuffle": True,
            "mode": "train",   # train, eval
            "data_root": None,
            "values_name": None,
            "indicators_train_name": None,
            "indicators_eval_name": None,
            "network_name": None
        }
