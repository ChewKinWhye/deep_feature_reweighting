import torch
import torchvision

import numpy as np
import os
import tqdm
import argparse
import sys
import json
from functools import partial
from wb_data import WaterBirdsDataset, get_loader, wb_transform, log_data
from argparse import Namespace
from BalancedOptimizer import BalancedOptimizer
from utils import MultiTaskHead, Discriminator
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from utils import update_dict, get_results, write_dict_to_tb
from utils import feature_reg_loss_specific, contrastive_loss, retain_feature_loss, coral_loss, correlation_loss, \
    MTL_Loss
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from methods.weight_methods import WeightMethods
from cmnist import get_cmnist
from torch.utils.data import Dataset, DataLoader


def parse_args():
    # --- Parser Start ---
    parser = argparse.ArgumentParser(description="Train model on artificial dataset")
    # Data Directory
    parser.add_argument("--dataset", type=str, default="cmnist",
                        help="Which dataset to use: [cmnist, mcdominoes]")
    # Output Directory
    parser.add_argument(
        "--output_dir", type=str,
        default="logs/",
        help="Output directory")

    # Model
    parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained model")

    # Data
    parser.add_argument("--val_size", type=int, default=200, help="Size of validation dataset")
    parser.add_argument("--spurious_strength", type=float, default=1, help="Strength of spurious correlation")

    # Training
    parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)

    # Different methods and combination of methods
    parser.add_argument("--method", type=int, default=0, help="Which method to use")
    # Method 0: Normal ERM
    # Method 1: Only balanced dataset
    # Method 2: MTL
    # Method 3: Balanced Optimizer
    # Scale of the methods

    args = parser.parse_args()
    # --- Parser End ---
    return args


# parameters in config overwrites the parser arguments
def main(args):
    # --- Logger Start ---
    print('Preparing directory %s' % args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'command.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        args_json = json.dumps(vars(args))
        f.write(args_json)
    set_seed(args.seed)
    logger = Logger(os.path.join(args.output_dir, 'log.txt'))
    # --- Logger End ---

    # --- Data Start ---
    if args.method == 0:
        indicies_val = np.arange(args.val_size)
        indicies_target = None
    else:
        indicies = np.arange(args.val_size)
        np.random.shuffle(indicies)
        # First half for val
        indicies_val = indicies[:len(indicies) // 2]
        # Second half for target
        indicies_target = indicies[len(indicies) // 2:]

    # Obtain trainset, valset_target, and testset_dict
    if args.dataset == "cmnist":
        target_resolution = (32, 32)
        trainset, valset_target, testset_dict = get_cmnist(target_resolution, args.val_size, args.spurious_strength, indicies_val, indicies_target)
    num_classes, num_places = testset_dict["Test"].n_classes, testset_dict["Test"].n_places

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    # For method 1, the training dataset is the balanced dataset
    if args.method == 1:
        train_loader = DataLoader(valset_target, shuffle=True, **loader_kwargs)
    else:
        train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)

    test_loader_dict = {}
    for test_name, testset_v in testset_dict.items():
        test_loader_dict[test_name] = DataLoader(testset_v, shuffle=False, **loader_kwargs)

    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    log_data(logger, trainset, testset_dict['Test'], testset_dict['Validation'], get_yp_func=get_yp_func)
    # --- Data End ---

    # --- Model Start ---
    model = torchvision.models.resnet18(pretrained=args.pretrained_model)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, num_classes)
    model.cuda()

    if args.method == 3:
        optimizer = BalancedOptimizer(model.parameters(), lr=args.init_lr, momentum=args.momentum_decay,
                                      weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss()
    # --- Model End ---

    # For method 7: NashMTL
    weight_methods_parameters = {'update_weights_every': 1, 'optim_niter': 20}
    weight_method = WeightMethods("nashmtl", n_tasks=2, device=torch.device('cuda'), **weight_methods_parameters)

    # --- Train Start ---
    best_worst_acc = 0

    for epoch in range(args.num_epochs):
        model.train()
        # Track metrics
        loss_meter = AverageMeter()
        method_loss_meter = AverageMeter()

        for batch in tqdm.tqdm(train_loader):
            # Data
            x, y, g, p, idxs = batch
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            optimizer.zero_grad()

            # --- Methods Start ---
            logits = model(x)
            loss = criterion(logits, y)

            # Normal ERM
            if args.method == 0 or args.method == 1:
                method_loss = 0
                loss.backward()
            # Methods with method loss
            else:
                random_indices = np.random.choice(len(valset_target), args.batch_size, replace=False)
                x_b, y_b, _, _, _ = valset_target.__getbatch__(random_indices)
                x_b, y_b = x_b.cuda(), y_b.type(torch.LongTensor).cuda()

                if args.method == 2:
                    logits_b = model(x_b)
                    method_loss = criterion(logits_b, y_b)
                    losses = torch.stack((loss, method_loss,))
                    # print(losses)
                    weight_method.backward(losses=losses, shared_parameters=list(model.parameters()),
                                           task_specific_parameters=None, last_shared_parameters=None,
                                           representation=None,)
                # Balanced Optimizer
                elif args.method == 3:
                    loss.backward()
                    optimizer.store_gradients("main")
                    optimizer.zero_grad()

                    logits_b = model(x_b)
                    method_loss = criterion(logits_b, y_b)
                    method_loss.backward()
                    optimizer.store_gradients("balanced")

            optimizer.step()
            method_loss_meter.update(method_loss, x.size(0))
            loss_meter.update(loss, x.size(0))
            # --- Methods Ends ---

        if args.scheduler:
            scheduler.step()

        # Save results
        logger.write(f"Epoch {epoch}\t ERM Loss: {loss_meter.avg}\t Method Loss: {method_loss_meter.avg}\n")

        # Evaluation
        # Iterating over datasets we test on
        for test_name, test_loader in test_loader_dict.items():
            results = evaluate(model, test_loader, get_yp_func, args.multitask)
            minority_acc = []
            majority_acc = []
            for y in range(num_classes):
                for p in range(num_places):
                    if y == p:
                        majority_acc.append(results[f"accuracy_{y}_{p}"])
                    else:
                        minority_acc.append(results[f"accuracy_{y}_{p}"])
            minority_acc = sum(minority_acc) / len(minority_acc)
            majority_acc = sum(majority_acc) / len(majority_acc)
            logger.write(f"Minority {test_name} accuracy: {minority_acc}")
            logger.write(f"Majority {test_name} accuracy: {majority_acc}")

        # Save best model based on worst group accuracy
        if minority_acc > best_worst_acc:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pt'))
            best_worst_acc = minority_acc

        logger.write('\n')

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
    # --- Train End ---

    logger.write(f'Best validation worst-group accuracy: {best_worst_acc}')
    logger.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
