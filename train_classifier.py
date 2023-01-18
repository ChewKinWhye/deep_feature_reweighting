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
import math
from wb_data import WaterBirdsDataset, get_loader, get_transform_cub, log_data
from argparse import Namespace
from ray import tune

from utils import MultiTaskHead, Discriminator
from utils import Logger, AverageMeter, set_seed, evaluate, get_y_p
from utils import update_dict, get_results, write_dict_to_tb
from utils import kaiming_init, normal_init, get_embed, permute_dims
from utils import feature_reg_loss_specific, contrastive_loss, retain_feature_loss, coral_loss, correlation_loss, MTL_Loss
from visualization import visualize_activations
from torchsummary import summary
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from methods.weight_methods import WeightMethods

VAL_SIZE = 1199

def parse_args():
        # --- Parser Start ---
    parser = argparse.ArgumentParser(description="Train model on waterbirds data")
    # Data Directory
    parser.add_argument(
        "--data_dir", type=str,
        default="/home/bizon/Desktop/KinWhye/BalancingGroups/data/waterbirds/waterbird_complete95_forest2water2",
        help="Train dataset directory")
    # Output Directory
    parser.add_argument(
        "--output_dir", type=str,
        default="logs/",
        help="Output directory")

    # Model
    parser.add_argument("--pretrained_model", action='store_true', help="Use pretrained model")

    # Data
    parser.add_argument("--reweight_classes", action='store_true', help="Reweight classes")
    parser.add_argument("--reweight_groups", action='store_true', help="Reweight groups")
    parser.add_argument("--augment_data", action='store_true', help="Train data augmentation")
    parser.add_argument("--multitask", action='store_true', help="Predict label and group")

    # Training
    parser.add_argument("--scheduler", action='store_true', help="Learning rate scheduler")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum_decay", type=float, default=0.9)
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)

    # Different methods and combination of methods
    parser.add_argument("--method", type=int, default=0, help="Which method to use")
    
    # Method 0: Normal ERM
    # Method 1: Contrast All
    # Method 2: Contrast Tenth
    # Method 3: Contrast 1/100
    # Method 4: Coral
    # Method 5: Conditional Coral
    # Method 6: Conditional Independence via Correlation Matrix
    # Method 7: MTL
    # Scale of the methods
    parser.add_argument("--contrast_temperature", type=float, default=0.5, help="contrast the other half of the feature space inversely")
    parser.add_argument("--method_scale", type=float, default=0.1, help="Scale of feature regularization")
    parser.add_argument("--method_dataset", type=float, default=0.1, help="Scale of feature regularization")


    args = parser.parse_args()
    assert args.reweight_groups + args.reweight_classes <= 1
    # --- Parser End ---
    return args

# parameters in config overwrites the parser arguments
def main(config=None, args=None):
    if config is not None:
        args = vars(args)
        for key, value in config.items():
            args[key] = value
        args = Namespace(**args)
        print(args)
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
    splits = ["train", "test", "val"]
    basedir = args.data_dir
    target_resolution = (224, 224)
    train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=args.augment_data)
    test_transform = get_transform_cub(target_resolution=target_resolution, train=False, augment_data=args.augment_data)

    # For methods that require the target dataset, we split the validation set into two
    if args.method in [1, 2, 3, 4, 5, 7]:
        indicies = np.arange(VAL_SIZE)
        np.random.shuffle(indicies)
        # First half for val
        indicies_val = indicies[:len(indicies)//2]
        # Second half for target
        indicies_target = indicies[len(indicies)//2:]
        # Save validation indicies just in case it is needed
        with open(os.path.join(args.output_dir, "indicies_val.npy"), "wb") as f:
            np.save(f, indicies_val)
        # Obtain target set
        valset_target = WaterBirdsDataset(basedir=basedir, split="val", transform=train_transform, indicies=indicies_target)
    # Otherwise we simply use the entire validation dataset
    else:
        indicies_val = np.arange(VAL_SIZE)
        valset_target = None

    trainset = WaterBirdsDataset(basedir=basedir, split="train", transform=train_transform)
    testset_dict = {
        'Test': WaterBirdsDataset(basedir=basedir, split="test", transform=test_transform),
        'Validation': WaterBirdsDataset(basedir=basedir, split="val", transform=test_transform, indicies=indicies_val),
    }
    # For methods that use the validation dataset, the validation dataset is split into two, one for training and the other for tuning

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = get_loader(
        trainset, train=True, reweight_groups=args.reweight_groups,
        reweight_classes=args.reweight_classes, reweight_places=False, **loader_kwargs)
    test_loader_dict = {}
    for test_name, testset_v in testset_dict.items():
        test_loader_dict[test_name] = get_loader(
            testset_v, train=False, reweight_groups=None,
            reweight_classes=None, reweight_places=None, **loader_kwargs)

    get_yp_func = partial(get_y_p, n_places=trainset.n_places)
    log_data(logger, trainset, testset_dict['Test'], get_yp_func=get_yp_func)
    # --- Data End ---

    # --- Model Start ---
    n_classes = trainset.n_classes
    model = torchvision.models.resnet50(pretrained=args.pretrained_model)

    d = model.fc.in_features
    if not args.multitask:
        model.fc = torch.nn.Linear(d, n_classes)
    else:
        model.fc = MultiTaskHead(d, [n_classes, trainset.n_places])
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
        acc_groups = {g_idx : AverageMeter() for g_idx in range(trainset.n_groups)}
        if args.multitask:
            acc_place_groups = {g_idx: AverageMeter() for g_idx in range(trainset.n_groups)}

        for batch in tqdm.tqdm(train_loader):
            # Data
            x, y, g, p, idxs = batch
            x, y, p = x.cuda(), y.cuda(), p.cuda()
            method_loss = 0
            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            if args.multitask:
                logits, logits_place = logits
                loss = criterion(logits, y) + criterion(logits_place, p)
                update_dict(acc_place_groups, p, g, logits_place)
            else:
                loss = criterion(logits, y)

            # --- Methods Start ---
            # Additional loss to regularize feature space based on method chosen
            # Contrast
            if args.method in [1, 2, 3]:
                random_indices = np.random.choice(len(valset_target), args.batch_size, replace=False)
                x_b, y_b, _, p_b, _ = valset_target.__getbatch__(random_indices)
                x_b, y_b, p_b = x_b.cuda(), y_b.cuda(), p_b.cuda()
                method_loss = (contrastive_loss(model, x_b, y_b, p_b, args.contrast_temperature, args.method) * args.method_scale)
            elif args.method in [4, 5]:
                random_indices = np.random.choice(len(valset_target), args.batch_size, replace=False)
                target_batch = valset_target.__getbatch__(random_indices)
                method_loss = coral_loss(model, x, target_batch[0].cuda(), y, target_batch[1].cuda(), args.method) * args.method_scale
            elif args.method == 6:
                # For y==0
                method_loss = correlation_loss(model, x[torch.where(y==0)]) * args.method_scale
                # For y==1
                method_loss += correlation_loss(model, x[torch.where(y==1)]) * args.method_scale
            # --- Methods Ends ---
            if args.method == 7:
                random_indices = np.random.choice(len(valset_target), args.batch_size, replace=False)
                x_b, y_b, _, p_b, _ = valset_target.__getbatch__(random_indices)
                x_b, y_b, p_b = x_b.cuda(), y_b.type(torch.LongTensor).cuda(), p_b.cuda()
                method_loss = criterion(logits_b, y_b)
                logits_b = model(x_b)
                losses = torch.stack(
                    (
                        loss,
                        method_loss,
                    )
                )
                # print(losses)
                weight_method.backward(
                    losses=losses,
                    shared_parameters=list(model.parameters()),
                    task_specific_parameters=None,
                    last_shared_parameters=None,
                    representation=None,
                )           
            else:
                total_loss = loss + method_loss
                total_loss.backward()
            optimizer.step()
            method_loss_meter.update(method_loss, x.size(0))
            loss_meter.update(loss, x.size(0))
            update_dict(acc_groups, y, g, logits)

        if args.scheduler:
            scheduler.step()
        
        # Save results
        logger.write(f"Epoch {epoch}\t ERM Loss: {loss_meter.avg}\t Method Loss: {method_loss_meter}\n")
        try:
            results = get_results(acc_groups, get_yp_func)
            logger.write(f"Train results \n")
            logger.write(str(results) + "\n")
        except:
            print("Zero Division")
        tag = ""
        if args.multitask:
            results_place = get_results(acc_place_groups, get_yp_func)
            logger.write(f"Train place prediction results \n")
            logger.write(str(results_place) + "\n")

        # Evaluation
        if epoch % args.eval_freq == 0:
            # Iterating over datasets we test on
            for test_name, test_loader in test_loader_dict.items():
                results = evaluate(model, test_loader, get_yp_func, args.multitask)
                if args.multitask:
                    results, _ = results
                logger.write(f"{test_name} results \n")
                logger.write(str(results))
                logger.write('\n')
            # Save best model based on worst group accuracy
            worst_val_acc = min(results["accuracy_0_0"], results["accuracy_0_1"], results["accuracy_1_0"], results["accuracy_1_1"])
            if worst_val_acc > best_worst_acc:
                torch.save(
                    model.state_dict(), os.path.join(args.output_dir, 'best_checkpoint.pt'))
                best_worst_acc = worst_val_acc
            
        logger.write('\n')

    torch.save(model.state_dict(), os.path.join(args.output_dir, 'final_checkpoint.pt'))
    # --- Train End ---

    logger.write(f'Best validation worst-group accuracy: {best_worst_acc}')
    logger.write('\n')


if __name__ == "__main__":
    args = parse_args()
    main(config=None, args=args)
