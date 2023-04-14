import torch
import torchvision
import numpy as np
import tqdm
import argparse
from functools import partial
from cmnist import get_cmnist
from mcdominoes import get_mcdominoes
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from utils import set_seed, evaluate, get_y_p, get_embed
from wb_data import WaterBirdsDataset, wb_transform

C_OPTIONS = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 2.5, 3]
REG = "l1"

def dfr_on_validation_tune(
        all_embeddings, all_y, all_p, preprocess=True, num_retrains=30):

    worst_accs = {}
    for i in range(num_retrains):
        # Only use validation data for the last layer retraining
        x_val = all_embeddings["val"]
        y_val = all_y["val"]
        p_val = all_p["val"]
        n_val = len(x_val) // 2
        idx = np.arange(len(x_val))
        np.random.shuffle(idx)

        # Split validation data into half
        x_train = x_val[idx[n_val:]]
        y_train = y_val[idx[n_val:]]

        x_val = x_val[idx[:n_val]]
        y_val = y_val[idx[:n_val]]
        p_val = p_val[idx[:n_val]]
        if preprocess:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_val = scaler.transform(x_val)

        class_weight = {0: 1., 1: 1.}
        for c in C_OPTIONS:
            logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
            logreg.fit(x_train, y_train)
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
    print(ks, vs)
    best_hypers = ks[np.argmax(vs)]
    return best_hypers


def dfr_on_validation_eval(
        c, all_embeddings, all_y, all_p, num_retrains=30, preprocess=True):
    coefs, intercepts = [], []
    if preprocess:
        scaler = StandardScaler()
        scaler.fit(all_embeddings["train"])

    for i in range(num_retrains):
        x_train = all_embeddings["val"]
        y_train = all_y["val"]

        if preprocess:
            x_train = scaler.transform(x_train)

        logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
        logreg.fit(x_train, y_train)
        coefs.append(logreg.coef_)
        intercepts.append(logreg.intercept_)

    x_test = all_embeddings["test"]
    y_test = all_y["test"]
    p_test = all_p["test"]
    # print(np.bincount(g_test))

    if preprocess:
        x_test = scaler.transform(x_test)
    logreg = LogisticRegression(penalty=REG, C=c, solver="liblinear")
    n_classes = np.max(y_train) + 1
    # the fit is only needed to set up logreg
    logreg.fit(x_train[:n_classes], np.arange(n_classes))
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
    print(minority_acc / minority_sum, majority_acc / majority_sum)

def parse_args():
    parser = argparse.ArgumentParser(description="Tune and evaluate DFR.")
    parser.add_argument("--dataset", type=str, default="cmnist",
                        help="Which dataset to use: [cmnist, mcdominoes]")
    parser.add_argument("--val_size", type=int, default=1000, help="Size of validation dataset")
    parser.add_argument("--spurious_strength", type=float, default=1, help="Strength of spurious correlation")
    parser.add_argument("--method", type=int, default=0, help="which method to retrain")
    parser.add_argument(
        "--ckpt_path", type=str, default=None, help="Checkpoint path")
    parser.add_argument(
        "--batch_size", type=int, default=100, required=False,
        help="Checkpoint path")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    return args


# parameters in config overwrites the parser arguments
def main(args):
    set_seed(args.seed)
    # --- Logger End ---

    # --- Data Start ---
    indicies_val = np.arange(args.val_size)
    if args.method != 0:
        indicies_target = []
    else:
        indicies_target = None

    # Obtain trainset, valset_target, and testset_dict
    if args.dataset == "cmnist":
        target_resolution = (32, 32)
        trainset, valset_target, testset_dict = get_cmnist(target_resolution, args.val_size, args.spurious_strength, indicies_val, indicies_target)
    elif args.dataset == "mcdominoes":
        target_resolution = (64, 32)
        trainset, valset_target, testset_dict = get_mcdominoes(target_resolution, args.val_size, args.spurious_strength, indicies_val, indicies_target)
    elif args.dataset == "waterbirds":
        data_dir = "/hpctmp/e0200920/waterbird_complete95_forest2water2"
        target_resolution = (224, 224)
        train_transform = wb_transform(target_resolution=target_resolution, train=True, augment_data=True)
        test_transform = wb_transform(target_resolution=target_resolution, train=False, augment_data=False)
        trainset = WaterBirdsDataset(basedir=data_dir, split="train", transform=train_transform)
        if indicies_target is not None:
            valset_target = WaterBirdsDataset(basedir=data_dir, split="val", transform=train_transform, indicies=indicies_target)
        else:
            valset_target = None
        testset_dict = {
            'Test': WaterBirdsDataset(basedir=data_dir, split="test", transform=test_transform),
            'Validation': WaterBirdsDataset(basedir=data_dir, split="val", transform=test_transform, indicies=indicies_val),
        }
        if args.spurious_strength == 1:
            group_counts = trainset.group_counts
            minority_groups = np.argsort(group_counts.numpy())[:2]
            idx = np.where(np.logical_and.reduce([trainset.group_array != g for g in minority_groups], initial=True))[0]
            trainset.y_array = trainset.y_array[idx]
            trainset.group_array = trainset.group_array[idx]
            trainset.confounder_array = trainset.confounder_array[idx]
            trainset.filename_array = trainset.filename_array[idx]
            trainset.metadata_df = trainset.metadata_df.iloc[idx]


    num_classes, num_places = testset_dict["Test"].n_classes, testset_dict["Test"].n_places

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}

    train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(
        testset_dict["Test"], shuffle=True, **loader_kwargs)
    val_loader = DataLoader(
        testset_dict["Validation"], shuffle=True, **loader_kwargs)

    # --- Data End ---

    # --- Model Start ---
    n_classes = trainset.n_classes
    model = torchvision.models.resnet18(pretrained=False)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, n_classes)
    model.load_state_dict(torch.load(
        args.ckpt_path
    ))
    model.cuda()
    model.eval()

    # --- Base Model Evaluation Start ---
    print("Base Model Results")
    base_model_results = {}
    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    base_model_results["test"] = evaluate(model, test_loader, silent=True)
    base_model_results["val"] = evaluate(model, val_loader, silent=True)
    base_model_results["train"] = evaluate(model, train_loader, silent=True)
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
    c = dfr_on_validation_tune(all_embeddings, all_y, all_p)

    dfr_val_results["best_hypers"] = c
    print("Best C value:", c)

    print("Validation Test")
    dfr_on_validation_eval(c, all_embeddings, all_y, all_p)

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
