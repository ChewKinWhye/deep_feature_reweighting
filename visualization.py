"""Evaluate DFR on spurious correlations datasets."""

import torch
import torchvision
import tqdm
import argparse
from wb_data import WaterBirdsDataset, WaterBirdsDataset2, get_loader, get_transform_cub, log_data
import matplotlib.pyplot as plt
import numpy as np
import os
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
from PIL import Image

def get_embed(m, x):
    results = []
    x = m.conv1(x)
    x = m.bn1(x)
    x = m.relu(x)
    x = m.maxpool(x)
    results.append(x)
    x = m.layer1(x)
    results.append(x)
    x = m.layer2(x)
    results.append(x)
    x = m.layer3(x)
    results.append(x)
    x = m.layer4(x)
    x = m.avgpool(x)
    results.append(x)
    x = torch.flatten(x, 1)
    return x, results


def visualize_activations(savedir, epoch, model):
    model.eval()
    target_resolution = (224, 224)
    test_transform = get_transform_cub(target_resolution=target_resolution,
                                       train=False, augment_data=False)

    loader_kwargs = {'batch_size': 32,
                     'num_workers': 4, 'pin_memory': True,
                     "reweight_places": None}

    testset = WaterBirdsDataset2(
        basedir="/home/bizon/Desktop/KinWhye/BalancingGroups/data/waterbirds/waterbird_complete95_forest2water2",
        segdir="/home/bizon/Desktop/KinWhye/BalancingGroups/data/segmentations", split="test", transform=test_transform)
    test_loader = get_loader(
        testset, train=False, reweight_groups=None, reweight_classes=None,
        **loader_kwargs)
    fg_activations = [[0]*64, [0]*256, [0]*512, [0]*1024, [0]*2048]
    bg_activations = [[0]*64, [0]*256, [0]*512, [0]*1024, [0]*2048]
    total_activations = [[0]*64, [0]*256, [0]*512, [0]*1024, [0]*2048]

    with torch.no_grad():
        for fg, bg, total in tqdm.tqdm(test_loader):
            fg, bg, total = fg.cuda(), bg.cuda(), total.cuda()
            # Store activations for foreground
            out, results = get_embed(model, fg)
            for num_layer in range(len(results)):
                layer_viz = torch.sum(results[num_layer], dim=0)
                layer_viz = layer_viz.data
                for num_filter, filter in enumerate(layer_viz):
                    fg_activations[num_layer][num_filter] += torch.mean(filter).cpu().item()

            # Store activations for background
            out, results = get_embed(model, bg)
            for num_layer in range(len(results)):
                layer_viz = torch.sum(results[num_layer], dim=0)
                layer_viz = layer_viz.data
                for num_filter, filter in enumerate(layer_viz):
                    bg_activations[num_layer][num_filter] += torch.mean(filter).cpu().item()

            # Store activations for total
            out, results = get_embed(model, total)
            for num_layer in range(len(results)):
                layer_viz = torch.sum(results[num_layer], dim=0)
                layer_viz = layer_viz.data
                for num_filter, filter in enumerate(layer_viz):
                    total_activations[num_layer][num_filter] += torch.mean(filter).cpu().item()
    layer = 4
    layer_ratio = []
    sensitivity = []
    for fg_activation, bg_activation, total_activation in zip(fg_activations[layer], bg_activations[layer], total_activations[layer]):
        bg_sensitivity = abs(total_activation-fg_activation)
        fg_sensitivity = abs(total_activation-bg_activation)
        sensitivity.append(bg_sensitivity+fg_sensitivity)
    sensitivity.sort()
    threshold = sensitivity[int(len(sensitivity)/10)]
    for fg_activation, bg_activation, total_activation in zip(fg_activations[layer], bg_activations[layer], total_activations[layer]):
        bg_sensitivity = abs(total_activation-fg_activation)
        fg_sensitivity = abs(total_activation-bg_activation)
        sensitivity.append(bg_sensitivity+fg_sensitivity)
        if bg_sensitivity+fg_sensitivity > threshold:
            layer_ratio.append(abs(bg_sensitivity-fg_sensitivity) / (bg_sensitivity+fg_sensitivity))
        else:
            layer_ratio.append(-1)
    dropped_ratio = [i for i in layer_ratio if i is not -1]
    n, bins, patches = plt.hist(x=dropped_ratio, bins='auto', color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "plots", f"epoch_{epoch}_layer_{layer}_ratio.png"))
    plt.clf()
    model.train()
    return sum(dropped_ratio)/len(dropped_ratio), layer_ratio

def plot_activations(model):
    target_resolution = (224, 224)
    test_transform = get_transform_cub(target_resolution=target_resolution,
                                       train=False, augment_data=False)
    testset = WaterBirdsDataset(basedir="/home/bizon/Desktop/KinWhye/BalancingGroups/data/waterbirds/waterbird_complete95_forest2water2",
                                split="test", transform=test_transform)
    loader_kwargs = {'batch_size': 1,
                     'num_workers': 4, 'pin_memory': True,
                     "reweight_places": None}
    test_loader = get_loader(
        testset, train=False, reweight_groups=None, reweight_classes=None,
        **loader_kwargs)
    with torch.no_grad():
        for x, y, g, p in tqdm.tqdm(test_loader):
            x = x.cuda()
            _, activations = get_embed(model, x)
            for num_layer in range(len(activations)):
                layer_viz = activations[num_layer][0]
                layer_viz = layer_viz.data
                for num_filter, filter in enumerate(layer_viz):
                    if (num_layer == 0 and num_filter == 29) \
                    or (num_layer == 1 and num_filter == 117)\
                    or (num_layer == 2 and num_filter == 41):
                        print(filter.cpu().numpy())
                        plt.imshow(filter.cpu().numpy(), cmap='gray')
                        plt.show()
                        plt.clf()

# 64, 256, 512, 1024, 2048
if __name__ == "__main__":
    # Load Model
    model = torchvision.models.resnet50(pretrained=False)
    d = model.fc.in_features
    model.fc = torch.nn.Linear(d, 2)
    model.load_state_dict(torch.load(
        "base_l1/final_checkpoint.pt"
    ))
    model.cuda()
    model.eval()

    # Obtain layer ratios
    average, layer_ratios = visualize_activations(savedir=None, epoch=None, model=model)
    print(f"Average Layer Ratio: {average}")
    layer_ratios = np.array(layer_ratios)
    sort_index = np.argsort(layer_ratios)
    sort_index = [i for i in sort_index if layer_ratios[i] != -1]

    print(f"Smallest Ratio {layer_ratios[sort_index[0]]} at index {sort_index[0]}")
    print(f"Largest Ratio {layer_ratios[sort_index[-1]]} at index {sort_index[-1]}")

    # Select features we want to visualize
    with torch.no_grad():
        model.fc.weight = torch.nn.Parameter(torch.zeros_like(model.fc.weight))
        model.fc.weight[:, sort_index[-1]] = torch.nn.Parameter(torch.ones_like(model.fc.weight[:, sort_index[-1]]))
    model.cuda()
    model.eval()

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    target_resolution = (224, 224)
    train_transform = get_transform_cub(target_resolution=target_resolution, train=True, augment_data=False)
    trainset = WaterBirdsDataset(basedir="/home/bizon/Desktop/KinWhye/BalancingGroups/data/waterbirds/waterbird_complete95_forest2water2", split="train", transform=train_transform)

    for target_y in [0, 1]:
        for target_p in [0, 1]:
            for i in [0, 1]:
                target = np.where((trainset.y_array == target_y) & (trainset.p_array == target_p))[0]
                x, y, g, p, rgb_img = trainset.__getitemrgb__(target[i])
                x = torch.unsqueeze(x, 0).cuda()
                grayscale_cam = cam(input_tensor=x, targets=None)
                # In this example grayscale_cam has only one image in the batch:
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                visualization = Image.fromarray(visualization)
                visualization.save(f"largest_{y}_{p}_{i}.jpg")
