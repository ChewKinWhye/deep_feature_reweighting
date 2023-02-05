import torch
import torchvision
import time
import numpy as np
import os
import tqdm
import argparse
import sys
import json
from functools import partial
from wb_data import WaterBirdsDataset, get_loader, wb_transform, log_data
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from cmnist import get_cmnist
from mcdominoes import get_mcdominoes
from torch.utils.data import Dataset, DataLoader
from GPM import ResNet18, get_model, set_model_, update_GPM, get_representation_matrix_ResNet18

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
    elif args.dataset == "mcdominoes":
        target_resolution = (64, 32)
        trainset, valset_target, testset_dict = get_mcdominoes(target_resolution, args.val_size, args.spurious_strength, indicies_val, indicies_target)

    num_classes, num_places = testset_dict["Test"].n_classes, testset_dict["Test"].n_places

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    # For method 1, the training dataset is the balanced dataset

    target_loader = DataLoader(valset_target, shuffle=True, **loader_kwargs)
    train_loader = DataLoader(trainset, shuffle=True, **loader_kwargs)

    test_loader_dict = {}
    for test_name, testset_v in testset_dict.items():
        test_loader_dict[test_name] = DataLoader(testset_v, shuffle=False, **loader_kwargs)

    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    log_data(logger, trainset, testset_dict['Test'], testset_dict['Validation'], get_yp_func=get_yp_func)
    # --- Data End ---

    model = ResNet18([(0, 10), (1, 10)], 20)  # base filters: 20
    threshold = np.array([0.99] * 20)

    best_model = get_model(model)
    model.cuda()

    optimizer = torch.optim.SGD(
            model.parameters(), lr=args.init_lr, momentum=args.momentum_decay, weight_decay=args.weight_decay)
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_epochs)
    else:
        scheduler = None

    criterion = torch.nn.CrossEntropyLoss()
    # --- Model End ---

    # --- Train Start ---
    best_worst_acc = 0

    for epoch in range(args.num_epochs):
        model.train()
        # Track metrics
        loss_meter = AverageMeter()
        method_loss_meter = AverageMeter()
        start = time.time()
        for batch in tqdm.tqdm(target_loader, disable=True):
            # Data
            x, y, g, p, idxs = batch
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            optimizer.zero_grad()

            # --- Methods Start ---
            logits = model(x)
            loss = criterion(logits[0], y)

            method_loss = 0
            loss.backward()
            optimizer.step()
            method_loss_meter.update(method_loss, x.size(0))
            loss_meter.update(loss, x.size(0))
            # --- Methods Ends ---

        if args.scheduler:
            scheduler.step()

        # Save results
        logger.write(f"Epoch {epoch}\t ERM Loss: {loss_meter.avg:.3f}\t Method Loss: {method_loss_meter.avg:.3f}\t Time Taken: {time.time()-start:.3f}\n")

        # Evaluation
        # Iterating over datasets we test on
        for test_name, test_loader in test_loader_dict.items():
            results = evaluate(model, test_loader, get_yp_func, GPM=True)
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
            logger.write(f"Minority {test_name} accuracy: {minority_acc:.3f}\t")
            logger.write(f"Majority {test_name} accuracy: {majority_acc:.3f}\n")

        # Save best model based on worst group accuracy
        if minority_acc > best_worst_acc:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pt'))
            best_worst_acc = minority_acc
            best_model = get_model(model)

        logger.write('\n')

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
    # --- Train End ---

    logger.write(f'Best validation worst-group accuracy: {best_worst_acc}')
    logger.write('\n')
    set_model_(model, best_model)
    mat_list = get_representation_matrix_ResNet18(model, valset_target)
    feature_list = update_GPM(model, mat_list, threshold, feature_list=[])

    feature_mat = []
    # Projection Matrix Precomputation
    for i in range(len(feature_list)):
        Uf = torch.Tensor(np.dot(feature_list[i], feature_list[i].transpose())).cuda()
        print('Layer {} - Projection Matrix shape: {}'.format(i + 1, Uf.shape))
        feature_mat.append(Uf)

    for epoch in range(args.num_epochs):
        model.train()
        # Track metrics
        loss_meter = AverageMeter()
        method_loss_meter = AverageMeter()
        start = time.time()
        for batch in tqdm.tqdm(train_loader, disable=True):
            # Data
            x, y, g, p, idxs = batch
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            optimizer.zero_grad()

            # --- Methods Start ---
            logits = model(x)
            loss = criterion(logits[1], y)

            method_loss = 0
            loss.backward()
            kk = 0
            for k, (m, params) in enumerate(model.named_parameters()):
                if len(params.size()) == 4:
                    sz = params.grad.data.size(0)
                    params.grad.data = params.grad.data - torch.mm(params.grad.data.view(sz, -1), feature_mat[kk]).view(params.size())
                    kk += 1
                elif len(params.size()) == 1:
                    params.grad.data.fill_(0)

            optimizer.step()
            method_loss_meter.update(method_loss, x.size(0))
            loss_meter.update(loss, x.size(0))
            # --- Methods Ends ---

        if args.scheduler:
            scheduler.step()

        # Save results
        logger.write(f"Epoch {epoch}\t ERM Loss: {loss_meter.avg:.3f}\t Method Loss: {method_loss_meter.avg:.3f}\t Time Taken: {time.time()-start:.3f}\n")

        # Evaluation
        # Iterating over datasets we test on
        for test_name, test_loader in test_loader_dict.items():
            results = evaluate(model, test_loader, get_yp_func, GPM=1)
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
            logger.write(f"Minority {test_name} accuracy: {minority_acc:.3f}\t")
            logger.write(f"Majority {test_name} accuracy: {majority_acc:.3f}\n")

        # Save best model based on worst group accuracy
        if minority_acc > best_worst_acc:
            torch.save(
                model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pt'))
            best_worst_acc = minority_acc
            best_model = get_model(model)

        logger.write('\n')

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
    # --- Train End ---

    logger.write(f'Best validation worst-group accuracy: {best_worst_acc}')
    logger.write('\n')

if __name__ == "__main__":
    args = parse_args()
    main(args=args)
