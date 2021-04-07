import yaml
import torch
import argparse

from src import Trainer, Evaluator, DataIterator, NET3


class Main:
    def __init__(self, model_configs_path, run_configs_path, mode, task="missing"):
        """
        mode: "train", "eval", "train-eval"
        task: "missing", "future"
        """
        self.configs_model = yaml.safe_load(open(model_configs_path))
        self.configs_run = yaml.safe_load(open(run_configs_path))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = NET3(self.configs_model)
        self.model = self.model.to(self.device)

        self.mode = mode
        self.task = task

    def run(self):
        if self.mode == "train":
            self.train()
        elif self.mode == "eval":
            return self.evaluate()
        elif self.mode == "train-eval":
            self.train(is_eval=True)

    def train(self, is_eval=False):
        data_iterator = DataIterator(configs=self.configs_run["TrainIterator"])
        trainer = Trainer(configs=self.configs_run["Trainer"],
                          model=self.model,
                          iterator=data_iterator,
                          is_eval=is_eval)
        trainer.train()

    def evaluate(self):
        data_iterator = DataIterator(configs=self.configs_run["EvalIterator"])
        evaluator = Evaluator(configs=self.configs_run["Evaluator"],
                              model=self.model,
                              iterator=data_iterator,
                              task=self.task)
        return evaluator.eval()


parser = argparse.ArgumentParser()
parser.add_argument("-cm", "--config_model", type=str, required=True)
parser.add_argument("-cr", "--config_run", type=str, required=True)
parser.add_argument("-m", "--mode", type=str, required=True)
parser.add_argument("-t", "--task", type=str, required=True)
args = parser.parse_args()

main = Main(model_configs_path=args.config_model, run_configs_path=args.config_run, mode=args.mode, task=args.task)
main.run()
