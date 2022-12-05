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
from visualization import visualize_activations

# Extract embeddings
def get_embed(m, x):
    # m = torch.nn.Sequential(*list(m.children())[:-3])
    # print(m)
    # x = m(x)
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    x = m.layer1(x)
    x = m.layer2(x)
    x = m.layer3(x)

    x = m.layer4[0](x)
    x = m.layer4[1](x)
    # x = m.layer4[2].conv1(x)
    # x = m.layer4[2].bn1(x)
    # x = m.layer4[2].conv2(x)
    # x = m.layer4[2].bn2(x)
    x = m.layer4[2].relu(x)
    # x = m.layer4(x)
    x = m.avgpool(x)
    x = torch.flatten(x, 1)

    return x

np.set_printoptions(threshold=sys.maxsize)

# WaterBirds
C_OPTIONS = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
CLASS_WEIGHT_OPTIONS = [1., 2., 3., 10., 100., 300., 1000.]
# CelebA
REG = "l1"

CLASS_WEIGHT_OPTIONS = [{0: 1, 1: w} for w in CLASS_WEIGHT_OPTIONS] + [
        {0: w, 1: 1} for w in CLASS_WEIGHT_OPTIONS]


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
    "--batch_size", type=int, default=100, required=False,
    help="Checkpoint path")
parser.add_argument(
    "--balance_dfr_val", type=bool, default=True, required=False,
    help="Subset validation to have equal groups for DFR(Val)")
parser.add_argument(
    "--notrain_dfr_val", type=bool, default=True, required=False,
    help="Do not add train data for DFR(Val)")
args = parser.parse_args()


def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True,
        balance_val=False, add_train=True, num_retrains=30):

    worst_accs = {}
    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1

        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        x_valtrain = x_val[idx[n_val:]]
        y_valtrain = y_val[idx[n_val:]]
        g_valtrain = g_val[idx[n_val:]]

        n_groups = np.max(g_valtrain) + 1
        g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_valtrain = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
            y_valtrain = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
            g_valtrain = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        n_train = len(x_valtrain) if add_train else 0

        x_train = np.concatenate([all_embeddings["train"][:n_train], x_valtrain])
        y_train = np.concatenate([all_y["train"][:n_train], y_valtrain])
        g_train = np.concatenate([all_g["train"][:n_train], g_valtrain])
        # print(np.bincount(g_train))
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)


        cls_w_options = [{0: 1., 1: 1.}]
        for c in C_OPTIONS:
            for class_weight in cls_w_options:
                logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                            class_weight=class_weight)
                logreg.fit(x_train, y_train)
                preds_val = logreg.predict(x_val)
                group_accs = np.array(
                    [(preds_val == y_val)[g_val == g].mean()
                     for g in range(n_groups)])
                worst_acc = np.min(group_accs)
                if i == 0:
                    worst_accs[c, class_weight[0], class_weight[1]] = worst_acc
                else:
                    worst_accs[c, class_weight[0], class_weight[1]] += worst_acc
                # print(c, class_weight[0], class_weight[1], worst_acc, worst_accs[c, class_weight[0], class_weight[1]])
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        c, w1, w2, all_embeddings, all_y, all_g, num_retrains=20,
        preprocess=True, balance_val=False, add_train=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["train"])

    for i in range(num_retrains):
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1
        g_idx = [np.where(g_val == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_val = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_val = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_val = np.concatenate([g_val[g[:min_g]] for g in g_idx])

        n_train = len(x_val) if add_train else 0
        train_idx = np.arange(len(all_embeddings["train"]))
        np.random.shuffle(train_idx)
        train_idx = train_idx[:n_train]

        x_train = np.concatenate(
            [all_embeddings["train"][train_idx], x_val])
        y_train = np.concatenate([all_y["train"][train_idx], y_val])
        g_train = np.concatenate([all_g["train"][train_idx], g_val])
        # print(np.bincount(g_train))
        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                    class_weight={0: w1, 1: w2})
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    g_test = all_g["test"]
    # print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                class_weight={0: w1, 1: w2})
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    preds_test = logreg.predict(x_test)
    preds_train = logreg.predict(x_train)
    n_groups = np.max(g_train) + 1
    test_accs = [(preds_test == y_test)[g_test == g].mean()
                 for g in range(n_groups)]
    test_mean_acc = (preds_test == y_test).mean()
    train_accs = [(preds_train == y_train)[g_train == g].mean()
                  for g in range(n_groups)]
    return test_accs, test_mean_acc, train_accs, logreg.coef_

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
print("Base Model")
base_model_results = {}
get_yp_func = partial(get_y_p, n_places=trainset.n_places)
base_model_results["test"] = evaluate(model, test_loader, get_yp_func)
base_model_results["val"] = evaluate(model, val_loader, get_yp_func)
base_model_results["train"] = evaluate(model, train_loader, get_yp_func)
print(base_model_results)
print()

model.eval()

all_embeddings = {}
all_y, all_p, all_g = {}, {}, {}
for name, loader in [("train", train_loader), ("test", test_loader), ("val", val_loader)]:
    all_embeddings[name] = []
    all_y[name], all_p[name], all_g[name] = [], [], []
    for x, y, g, p in tqdm.tqdm(loader):
        with torch.no_grad():
            all_embeddings[name].append(get_embed(model, x.cuda()).detach().cpu().numpy())
            all_y[name].append(y.detach().cpu().numpy())
            all_g[name].append(g.detach().cpu().numpy())
            all_p[name].append(p.detach().cpu().numpy())
    all_embeddings[name] = np.vstack(all_embeddings[name])
    all_y[name] = np.concatenate(all_y[name])
    all_g[name] = np.concatenate(all_g[name])
    all_p[name] = np.concatenate(all_p[name])
print(get_embed(model, x.cuda()).detach().cpu().numpy().shape)

# DFR on validation
print("DFR on validation")
dfr_val_results = {}
c, w1, w2 = dfr_on_validation_tune(
    all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
dfr_val_results["best_hypers"] = (c, w1, w2)
print("Hypers:", (c, w1, w2))
test_accs, test_mean_acc, train_accs, coefs = dfr_on_validation_eval(
        c, w1, w2, all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val, add_train=not args.notrain_dfr_val)
dfr_val_results["test_accs"] = test_accs
dfr_val_results["train_accs"] = train_accs
dfr_val_results["test_worst_acc"] = np.min(test_accs)
dfr_val_results["test_mean_acc"] = test_mean_acc
print(dfr_val_results)
# disentanglement_ratio, layer_ratio = visualize_activations(None, None, model)
# # print(np.where(coefs[0] == 0)[0])
# # print(np.where(coefs[0] != 0)[0])
# layer_ratio_zero = []
# for i in np.where(coefs[0] == 0)[0]:
#     if layer_ratio[i] != -1:
#         layer_ratio_zero.append(layer_ratio[i])
# print(sum(layer_ratio_zero)/len(layer_ratio_zero))
#
# layer_ratio_non_zero = []
# for i in np.where(coefs[0] != 0)[0]:
#     if layer_ratio[i] != -1:
#         layer_ratio_non_zero.append(layer_ratio[i])
# print(sum(layer_ratio_non_zero)/len(layer_ratio_non_zero))

# DFR on train subsampled
# print("DFR on train subsampled")
# dfr_train_results = {}
# c, w1, w2 = dfr_train_subset_tune(
#     all_embeddings, all_y, all_g,
#     learn_class_weights=args.tune_class_weights_dfr_train)
# dfr_train_results["best_hypers"] = (c, w1, w2)
# print("Hypers:", (c, w1, w2))
# test_accs, test_mean_acc, train_accs = dfr_train_subset_eval(
#         c, w1, w2, all_embeddings, all_y, all_g)
# dfr_train_results["test_accs"] = test_accs
# dfr_train_results["train_accs"] = train_accs
# dfr_train_results["test_worst_acc"] = np.min(test_accs)
# dfr_train_results["test_mean_acc"] = test_mean_acc
# print(dfr_train_results)
# print()
#
#
# all_results = {}
# all_results["base_model_results"] = base_model_results
# all_results["dfr_val_results"] = dfr_val_results
# all_results["dfr_train_results"] = dfr_train_results
# print(all_results)
#
# with open(args.result_path, 'wb') as f:
#     pickle.dump(all_results, f)