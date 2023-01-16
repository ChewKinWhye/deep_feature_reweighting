"""Evaluate DFR on spurious correlations datasets."""

import torch
import torchvision

import numpy as np
import os
import tqdm
import argparse
import sys
from collections import defaultdict
import json
from functools import partial
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
# from train_classifier import feature_reg_loss_specific, feature_reg_loss_general
from utils import update_dict, get_results, write_dict_to_tb


parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
parser.add_argument(
    "--data_dir", type=str,
    default="/home/bizon/Desktop/KinWhye/BalancingGroups/data/waterbirds/waterbird_complete95_forest2water2",
    help="Train dataset directory")
parser.add_argument(
    "--result_path", type=str, default="logs/",
    help="Path to save results")
parser.add_argument(
    "--ckpt_path", type=str, default=None, help="Checkpoint path")
parser.add_argument(
    "--batch_size", type=int, default=32, required=False,
    help="Checkpoint path")
parser.add_argument(
    "--weight_decay", type=float, default=1e-1, required=False,
    help="Weight Decay")
parser.add_argument(
    "--method", type=int, default=0, required=False,
    help="Which method to use")
# Method 0: Normal ERM on validation
# Method 1: ERM on validation and train
# Method 2: ERM on validation and regularize gradients of train
parser.add_argument(
    "--method_scale", type=float, default=1, required=False,
    help="Scale of loss of method")

args = parser.parse_args()

## Load data
target_resolution = (224, 224)
train_transform = get_transform_cub(target_resolution=target_resolution,
                                    train=True, augment_data=False)
test_transform = get_transform_cub(target_resolution=target_resolution,
                                   train=False, augment_data=False)

trainset = WaterBirdsDataset(
    basedir=args.data_dir, split="train", transform=train_transform)
testset = WaterBirdsDataset(
    basedir=args.data_dir, split="test", transform=test_transform)
valset = WaterBirdsDataset(
    basedir=args.data_dir, split="val", transform=test_transform)

loader_kwargs = {'batch_size': args.batch_size,
                 'num_workers': 4, 'pin_memory': True,
                 "reweight_places": None}
train_loader = get_loader(
    trainset, train=True, reweight_groups=False, reweight_classes=False,
    **loader_kwargs)
test_loader = get_loader(
    testset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)
val_loader = get_loader(
    valset, train=False, reweight_groups=None, reweight_classes=None,
    **loader_kwargs)

# Load model
n_classes = trainset.n_classes
model = torchvision.models.resnet50(pretrained=False)
d = model.fc.in_features
model.fc = torch.nn.Linear(d, n_classes)
model.load_state_dict(torch.load(
    args.ckpt_path
))
model.cuda()
model.eval()

# Evaluate model
# print("Base Model")
# base_model_results = {}
get_yp_func = partial(get_y_p, n_places=trainset.n_places)
# base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
# base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
# base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
# print(base_model_results)
# print()


model.train()
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-3, momentum=0.9, weight_decay=args.weight_decay)
scheduler = None

criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    loss_meter = AverageMeter()
    acc_groups = {g_idx: AverageMeter() for g_idx in range(trainset.n_groups)}

    for batch in tqdm.tqdm(val_loader):
        x, y, g, p, _ = batch
        x, y, p = x.cuda(), y.cuda(), p.cuda()

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        print(y)
        if args.method == 1:
            # Sample a random training batch and add the loss
            random_indices = np.random.choice(len(trainset), args.batch_size, replace=False)
            x_t, y_t, _, _, _ = trainset.__getbatch__(random_indices)
            x_t, y_t = x_t.cuda(), torch.flatten(torch.Tensor(y_t)).cuda()
            logits = model(x_t)
            loss += (criterion(logits, y) * args.method_scale)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss, x.size(0))
        update_dict(acc_groups, y, g, logits)

    results = get_results(acc_groups, get_yp_func)
    print(f"Validation: {results}")
    # Iterating over datasets we test on
    results = evaluate(model, test_loader, get_yp_func)
    print(f"Test: {results}")
    results = evaluate(model, train_loader, get_yp_func)
    print(f"Train: {results}")
