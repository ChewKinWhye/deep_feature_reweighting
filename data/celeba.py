import os
from torchvision import transforms
import pandas as pd
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


class CELEBA(Dataset):
    def __init__(self, x, y, p, target_resolution):
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

        self.transform = transforms.Compose(
            [
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]
        p = self.confounder_array[idx]

        img = self.x[idx]
        img = self.transform(Image.open(img).convert("RGB"))

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
        return torch.stack(x_batch), torch.flatten(torch.Tensor(np.vstack(y_batch))), torch.flatten(
            torch.Tensor(np.vstack(g_batch))), \
               torch.flatten(torch.Tensor(np.vstack(p_batch))), torch.flatten(torch.Tensor(np.vstack(idx_batch)))



# This celeba considers the inverse problem, where the target feature is the gender and the spurious feature
# is the hair color. This setup makes the spurious feature easier to learn than the target feature
def get_celeba(target_resolution, VAL_SIZE, spurious_strength, data_dir, seed, indicies_val=None, indicies_target=None):
    root = os.path.join(data_dir, "img_align_celeba")
    metadata = os.path.join(data_dir, "metadata_celeba.csv")
    df = pd.read_csv(metadata)

    # Train dataset
    train_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["tr"])]
    train_df_x = train_df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
    # y==0 is non-blonde and y==1 is blonde
    train_df_y = train_df["y"].tolist()
    train_df_a = np.array(train_df["a"].tolist())
    train_set = CELEBA(train_df_x, train_df_y, train_df_a, target_resolution)
    print(train_set.group_counts)
    val_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["va"])]
    val_df_x = val_df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
    val_df_y = val_df["y"].tolist()
    val_df_a = np.array(val_df["a"].tolist())
    val_dataset = CELEBA(val_df_x, val_df_y, val_df_a, target_resolution)


    test_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["te"])]
    test_df_x = test_df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
    test_df_y = test_df["y"].tolist()
    test_df_a = np.array(test_df["a"].tolist())
    test_dataset = CELEBA(test_df_x, test_df_y, test_df_a, target_resolution)

    testset_dict = {
        'Test': test_dataset,
        'Validation': val_dataset,
    }

