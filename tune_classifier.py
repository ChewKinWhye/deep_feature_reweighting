from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from train_classifier import parse_args
import ray
from train_classifier import main



def tune_params(num_samples, max_num_epochs, gpus_per_trial):
    args = parse_args()
    config = {
        "weight_decay": tune.choice([1e-2, 1e-3, 1e-4]),
        "batch_size": tune.choice([16, 32, 64, 128]),
        "init_lr": tune.choice([1e-2, 1e-3, 1e-4]),
        "method_scale": tune.choice([0.1, 0.5, 1, 2]),
    }
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["accuracy", "training_iteration"])
    
    # main = tune.with_resources(main, {"gpu": 1})
    result = tune.run(
        partial(main, args=args),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial(metric="accuracy", mode="max", scope="all")
    print("Best trial config: {}".format(best_trial.config))
    print(best_trial)
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    tune_params(num_samples=5, max_num_epochs=5, gpus_per_trial=1)
