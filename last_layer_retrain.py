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
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p, get_embed
#from visualization import visualize_activations

def dfr_on_validation_tune(
        all_embeddings, all_y, all_g, preprocess=True,
        balance_val=False, num_retrains=30):

    worst_accs = {}
    for i in range(num_retrains):
        # Only use validation data for the last layer retraining
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        g_val = all_g["val"]
        n_groups = np.max(g_val) + 1
        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        # Split validation data into half
        x_valtrain = x_val[idx[n_val:]]
        y_valtrain = y_val[idx[n_val:]]
        g_valtrain = g_val[idx[n_val:]]

        n_groups = np.max(g_valtrain) + 1
        g_idx = [np.where(g_valtrain == g)[0] for g in range(n_groups)]
        min_g = np.min([len(g) for g in g_idx])
        for g in g_idx:
            np.random.shuffle(g)
        if balance_val:
            x_train = np.concatenate([x_valtrain[g[:min_g]] for g in g_idx])
            y_train = np.concatenate([y_valtrain[g[:min_g]] for g in g_idx])
            g_train = np.concatenate([g_valtrain[g[:min_g]] for g in g_idx])

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        g_val = g_val[idx[:n_val]]

        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

        class_weight = {0: 1., 1: 1.}
        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear",
                                        class_weight=class_weight)
            logreg.fit(x_train, y_train)
            preds_val = logreg.predict(x_val)
            group_accs = np.array(
                [(preds_val == y_val)[g_val == g].mean()
                    for g in range(n_groups)])
            worst_acc = np.min(group_accs)
            if i == 0:
                worst_accs[c] = worst_acc
            else:
                worst_accs[c] += worst_acc
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        c, all_embeddings, all_y, all_g, num_retrains=30, w1=1, w2=1,
        preprocess=True, balance_val=False):
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
            x_train = np.concatenate([x_val[g[:min_g]] for g in g_idx])
            y_train = np.concatenate([y_val[g[:min_g]] for g in g_idx])
            g_train = np.concatenate([g_val[g[:min_g]] for g in g_idx])

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

# --- Default Parameters Start ---
C_OPTIONS = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
REG = "l1"
# --- Default Parameters End ---

# --- Parser Start ---
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
parser.add_argument("--reduce_dimension", action='store_true', help="Whether the model has reduced dimension")

args = parser.parse_args()
# --- Parser End ---

## --- Data Start ---
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
# --- Data End ---

# --- Model Start ---
n_classes = trainset.n_classes
model = torchvision.models.resnet50(pretrained=False)
d = model.fc.in_features
if args.reduce_dimension:
    # reduce layer reduces the dimenion to 50 before the classification layer
    model.fc = nn.Sequential(
    nn.Linear(d, 50),
    nn.ReLU(),
    nn.Linear(50, n_classes),
    )
else:
    model.fc = torch.nn.Linear(d, n_classes)
model.reduce_dimension = args.reduce_dimension

model.load_state_dict(torch.load(
    args.ckpt_path
))
model.cuda()
model.eval()
# --- Model End ---

# --- Base Model Evaluation Start ---
print("Base Model Results")
base_model_results = {}
get_yp_func = partial(get_y_p, n_places=trainset.n_places)
base_model_results["test"] = evaluate(model, test_loader, get_yp_func, silent=True)
base_model_results["val"] = evaluate(model, val_loader, get_yp_func, silent=True)
base_model_results["train"] = evaluate(model, train_loader, get_yp_func, silent=True)
print(base_model_results)
# --- Base Model Evaluation End ---

# --- Extract Embeddings Start ---
print("Extract Embeddings")
model.eval()
all_embeddings = {}
all_y, all_p, all_g = {}, {}, {}
for name, loader in [("train", train_loader), ("test", test_loader), ("val", val_loader)]:
    all_embeddings[name] = []
    all_y[name], all_p[name], all_g[name] = [], [], []
    for x, y, g, p, idxs in tqdm.tqdm(loader, disable=True):
        with torch.no_grad():
            # print(get_embed(model, x.cuda()).detach().cpu().numpy().shape)
            emb = get_embed(model, x.cuda()).detach().cpu().numpy()
            # Only do retraining on subset of last layer features
            # dividing_point = emb.shape[1]//10 * 9
            # emb = emb[:, :dividing_point]

            all_embeddings[name].append(emb)
            all_y[name].append(y.detach().cpu().numpy())
            all_g[name].append(g.detach().cpu().numpy())
            all_p[name].append(p.detach().cpu().numpy())
    all_embeddings[name] = np.vstack(all_embeddings[name])
    all_y[name] = np.concatenate(all_y[name])
    all_g[name] = np.concatenate(all_g[name])
    all_p[name] = np.concatenate(all_p[name])
# print(f"Embedding Shape: {get_embed(model, x.cuda()).detach().cpu().numpy().shape}")
# --- Extract Embeddings End ---

# DFR on validation
print("Validation Tune")
dfr_val_results = {}
c = dfr_on_validation_tune(
    all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val)

dfr_val_results["best_hypers"] = c
print("Best C value:", c)

print("Validation Test")
test_accs, test_mean_acc, train_accs, coefs = dfr_on_validation_eval(
        c, all_embeddings, all_y, all_g,
    balance_val=args.balance_dfr_val)
dfr_val_results["test_accs"] = test_accs
dfr_val_results["train_accs"] = train_accs
dfr_val_results["test_worst_acc"] = np.min(test_accs)
dfr_val_results["test_mean_acc"] = test_mean_acc
# dividing_point = emb.shape[1]//10 * 9
# print(np.mean(np.abs(coefs[0][:dividing_point])))
# print(np.mean(np.abs(coefs[0][dividing_point:])))

print(dfr_val_results)

# with open("coefs.txt", "w") as f:
#     for coef in coefs[0]:
#         f.write(f"{coef} ")

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
