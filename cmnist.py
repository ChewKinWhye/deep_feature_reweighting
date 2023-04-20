import os
import copy
import math
import json
import random as rnd
import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as  pd
import torchvision.utils as vision_utils
from PIL import Image
import torchvision
from scipy.linalg import qr
import torchvision.transforms as transforms

from matplotlib.ticker import NullFormatter

def color_train_digits(X, Y, spurious_strength):
    res = []
    p = []
    for x, y in zip(X, Y):
        x = x / 255
        mask = x.view(28,28) > 0.1
        img = x.repeat(3, 1, 1)
        if rnd.random() <= spurious_strength:
            color_idx = y
        else:
            color_idx = rnd.randint(0, 9)
        p.append(color_idx)
        if color_idx == 0:
            img[0][mask] *= 0.5
        elif color_idx == 1:
            img[1][mask] *= 0.5
        elif color_idx == 2:
            img[2][mask] *= 0.5
        elif color_idx == 3:
            img[0][mask] *= 0.2
            img[1][mask] *= 0.2
        elif color_idx == 4:
            img[0][mask] *= 0.1
            img[2][mask] *= 0.1
        elif color_idx == 5:
            img[1][mask] *= 0.6
            img[2][mask] *= 0.
        elif color_idx == 6:
            img[1][mask] *= 0.3
            img[2][mask] *= 0.2
        elif color_idx == 7:
            img[0][mask] *= 0.
            img[2][mask] *= 0.6
        elif color_idx == 8:
            img[0][mask] *= 0.5
            img[1][mask] *= 0.2
        else:
            pass
        res.append(img.clip(0,1))
    res = torch.stack(res)
    return res, p


def color_test_digits(X, Y):
    res = []
    p = []
    for x, y in zip(X, Y):
        x = x / 255
        mask = x.view(28,28) > 0.1
        img = x.repeat(3, 1, 1)
        color_idx = rnd.randint(0, 9)
        p.append(color_idx)
        if color_idx == 0:
            img[0][mask] *= 0.5
        elif color_idx == 1:
            img[1][mask] *= 0.5
        elif color_idx == 2:
            img[2][mask] *= 0.5
        elif color_idx == 3:
            img[0][mask] *= 0.2
            img[1][mask] *= 0.2
        elif color_idx == 4:
            img[0][mask] *= 0.1
            img[2][mask] *= 0.1
        elif color_idx == 5:
            img[1][mask] *= 0.6
            img[2][mask] *= 0.
        elif color_idx == 6:
            img[1][mask] *= 0.3
            img[2][mask] *= 0.2
        elif color_idx == 7:
            img[0][mask] *= 0.
            img[2][mask] *= 0.6
        elif color_idx == 8:
            img[0][mask] *= 0.5
            img[1][mask] *= 0.2
        else:
            pass
        res.append(img.clip(0,1))
    res = torch.stack(res)
    return res, p


class CMNIST(Dataset):
    def __init__(self, x, y, p, target_resolution):
        scale = 28.0 / 32.0
        self.x = x
        self.y_array = np.array(y)
        self.p_array = np.array(p)
        self.confounder_array = np.array(p)
        self.n_classes = np.unique(self.y_array).size
        self.n_places = np.unique(self.confounder_array).size
        self.group_array = (self.y_array * self.n_places + self.confounder_array).astype('int')
        self.n_groups = self.n_classes * self.n_places
        self.group_counts = (
                torch.arange(self.n_groups).unsqueeze(1) == torch.from_numpy(self.group_array)).sum(1).float()
        self.y_counts = (
                torch.arange(self.n_classes).unsqueeze(1) == torch.from_numpy(self.y_array)).sum(1).float()
        self.p_counts = (
                torch.arange(self.n_places).unsqueeze(1) == torch.from_numpy(self.p_array)).sum(1).float()
        
        self.transform = transforms.Compose([
                                    transforms.Resize((int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
                                    transforms.CenterCrop(target_resolution),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                ])

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]
        img = self.x[idx]
        img = self.transform(img)
        
        return img, y, g, p, idx

    def __getbatch__(self, idxs):
        x_batch, y_batch, g_batch, p_batch, idx_batch = [], [], [], [], []
        for idx in idxs:
            x, y, g, p, idx = self.__getitem__(idx)
            x_batch.append(x)
            y_batch.append(y)
            g_batch.append(g)
            p_batch.append(p)
            idx_batch.append(idx)
        return torch.stack(x_batch), torch.flatten(torch.Tensor(np.vstack(y_batch))), torch.flatten(torch.Tensor(np.vstack(g_batch))), \
         torch.flatten(torch.Tensor(np.vstack(p_batch))), torch.flatten(torch.Tensor(np.vstack(idx_batch)))


def get_cmnist(target_resolution, VAL_SIZE, spurious_strength, indicies_val=None, indicies_target=None):
    # Train
    train_set = torchvision.datasets.MNIST('./data/mnist/', train=True, download=True)
    train_input = train_set.data.view(-1, 1, 28, 28).float()
    train_target = train_set.targets
    rand_perm = torch.randperm(len(train_input))
    train_input = train_input[rand_perm]
    train_target = train_target[rand_perm]

    # Train-Val Split
    val_input, val_target = train_input[:VAL_SIZE], train_target[:VAL_SIZE]
    if indicies_target is not None:
        balanced_input, balanced_target = val_input[indicies_target], val_target[indicies_target]
    val_input, val_target = val_input[indicies_val], val_target[indicies_val]
    train_input, train_target = train_input[VAL_SIZE:], train_target[VAL_SIZE:]

    # Test
    test_set = torchvision.datasets.MNIST('./data/mnist/', train=False, download=True)
    test_input = test_set.data.view(-1, 1, 28, 28).float()
    test_target = test_set.targets
    rand_perm = torch.randperm(len(test_input))
    test_input = test_input[rand_perm]
    test_target = test_target[rand_perm]

    # Color Images
    train_input, train_p = color_train_digits(train_input, train_target, spurious_strength)
    val_input, val_p = color_test_digits(val_input, val_target)
    test_input, test_p = color_test_digits(test_input, test_target)
    if indicies_target is not None:
        balanced_input, balanced_p = color_test_digits(balanced_input, balanced_target)
        balanced_dataset = CMNIST(balanced_input, balanced_target, balanced_p, target_resolution)
    else:
        balanced_dataset = None
    train_dataset = CMNIST(train_input, train_target, train_p, target_resolution)
    val_dataset = CMNIST(val_input, val_target, val_p, target_resolution)
    test_dataset = CMNIST(test_input, test_target, test_p, target_resolution)
    testset_dict = {
        'Test': test_dataset,
        'Validation': val_dataset,
    }

    return train_dataset, balanced_dataset, testset_dict
# # Plot Images
# fig = plt.figure(figsize=(6,6), dpi=100)
# grid_img = vision_utils.make_grid((train_input[:100]).cpu(), 
#                                   nrow=10, 
#                                   normalize=True, 
#                                   padding=0)
# plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
# plt.title('Train data')

# fig = plt.figure(figsize=(6,6), dpi=100)
# grid_img = vision_utils.make_grid((val_input[:100]).cpu(), 
#                                   nrow=10, 
#                                   normalize=True, 
#                                   padding=0)
# plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
# plt.title('Validation data')

# fig = plt.figure(figsize=(6,6), dpi=100)
# grid_img = vision_utils.make_grid((test_input[:100]).cpu(), 
#                                   nrow=10, 
#                                   normalize=True, 
#                                   padding=0)
# plt.imshow(grid_img.permute(1, 2, 0), interpolation='nearest')
# plt.title('Test data')
# plt.show()
