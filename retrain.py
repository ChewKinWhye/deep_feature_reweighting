import torch
import torchvision
import numpy as np
import tqdm
import argparse
from functools import partial
from data.mcdominoes import get_mcdominoes
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import set_seed, evaluate, get_y_p, get_embed, Logger
from data.wb_data import WaterBirdsDataset, get_loader, wb_transform, log_data, get_waterbirds

import pickle
import os


C_OPTIONS = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
REG = "l1"

def dfr_on_target_tune(
        all_embeddings, all_y, all_p, num_retrains=30):

    worst_accs = {}
    for i in range(num_retrains):
        # Only use validation data for the last layer retraining
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        p_val = all_p["val"]

        x_target = all_embeddings["target"]
        y_target = all_y["target"]
        p_target = all_p["target"]


        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
            logreg.fit(x_target, y_target)
            preds_val = logreg.predict(x_val)
            minority_acc, minority_sum, majority_acc, majority_sum = 0, 0, 0, 0
            correct = y_val == preds_val
            for x in range(len(y_val)):
                if y_val[x] == p_val[x]:
                    majority_acc += correct[x]
                    majority_sum += 1
                else:
                    minority_acc += correct[x]
                    minority_sum += 1
            if i == 0:
                worst_accs[c] = minority_acc / minority_sum
            else:
                worst_accs[c] += minority_acc / minority_sum
    ks, vs = list(worst_accs.keys()), list(worst_accs.values())
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_target_eval(
        c, all_embeddings, all_y, all_p, num_retrains=30):
    coefs, intercepts = [], []

    for i in range(num_retrains):
        x_target = all_embeddings["target"]
        y_target = all_y["target"]

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
        logreg.fit(x_target, y_target)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    p_test = all_p["test"]
    # print(np.bincount(g_test))

    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
    n_classes = np.max(y_target) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_target[:n_classes], np.arange(n_classes))
    logreg.coef_ = np.mean(coefs, axis=0)
    logreg.intercept_ = np.mean(intercepts, axis=0)
    preds_test = logreg.predict(x_test)
    minority_acc, minority_sum, majority_acc, majority_sum = 0, 0, 0, 0
    correct = y_test == preds_test
    for i in range(len(y_test)):
        if y_test[i] == p_test[i]:
            majority_acc += correct[i]
            majority_sum += 1
        else:
            minority_acc += correct[i]
            minority_sum += 1
    return minority_acc / minority_sum, majority_acc / majority_sum

def parse_args():
    parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
    parser.add_argument("--dataset", type=str, default="cmnist",
                        help="Which dataset to use: [cmnist, mcdominoes]")
    parser.add_argument("--val_size", type=int, default=1000, help="Size of validation dataset")
    parser.add_argument("--spurious_strength", type=float, default=1, help="Strength of spurious correlation")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_dir", type=str,
        default="logs/",
        help="Output directory")

    args = parser.parse_args()

    return args


# parameters in config overwrites the parser arguments
def main(args):
    logger = Logger(os.path.join(args.output_dir, 'log.txt'))

    set_seed(args.seed)
    # --- Logger End ---

    # Obtain trainset, valset_target, and testset_dict
    if args.dataset == "mcdominoes":
        target_resolution = (64, 32)
        train_set, target_set, test_set_dict = get_mcdominoes(target_resolution, args.val_target_size, args.spurious_strength,
                                                              args.data_dir, args.seed)
    elif args.dataset == "waterbirds":
        target_resolution = (224, 224)
        train_set, target_set, test_set_dict = get_waterbirds(target_resolution, args.val_target_size,
                                                              args.spurious_strength,
                                                              args.data_dir, args.seed)

    num_classes, num_places = test_set_dict["Test"].n_classes, test_set_dict["Test"].n_places

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}

    target_loader = DataLoader(
        target_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(
        test_set_dict["Validation"], shuffle=True, **loader_kwargs)
    test_loader = DataLoader(
        test_set_dict["Test"], shuffle=True, **loader_kwargs)
    # --- Data End ---

    # --- Model Start ---
    n_classes = train_set.n_classes
    model = torchvision.models.resnet18(pretrained=False)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)
    model.load_state_dict(torch.load(
        args.ckpt_path
    ))
    model.cuda()
    model.eval()

    # --- Base Model Evaluation Start ---
    logger.write("Base Model Results")
    base_model_results = {}
    base_model_results["test"] = evaluate(model, test_loader, silent=True)
    logger.write(base_model_results)
    # --- Base Model Evaluation End ---

    # --- Extract Embeddings Start ---
    logger.write("Extract Embeddings")
    model.eval()
    all_embeddings = {}
    all_y, all_p, all_g = {}, {}, {}
    for name, loader in [("target", target_loader), ("val", val_loader), ("test", test_loader)]:
        all_embeddings[name] = []
        all_y[name], all_p[name], all_g[name] = [], [], []
        for x, y, g, p, idxs in tqdm.tqdm(loader, disable=True):
            with torch.no_grad():
                emb = get_embed(model, x.cuda()).detach().cpu().numpy()
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
    logger.write("Target Tune")
    dfr_val_results = {}
    c = dfr_on_target_tune(all_embeddings, all_y, all_p)

    dfr_val_results["best_hypers"] = c
    logger.write(f"Best C value: {c}")

    logger.write("Test")
    minority_test_acc, majority_test_acc = dfr_on_target_eval(c, all_embeddings, all_y, all_p)
    logger.write(f"Majority Test Accuracy: {majority_test_acc}")
    logger.write(f"Minority Test Accuracy: {minority_test_acc}")

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
