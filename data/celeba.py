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

# TODO: Remove minority
class CELEBA(Dataset):
    def __init__(self, x, y, p):
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
                transforms.CenterCrop(178),
                transforms.Resize(224),
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
def get_celeba():
    data_path = "data"
    root = os.path.join(data_path, "celeba/img_align_celeba/")
    metadata = os.path.join(data_path, "metadata_celeba.csv")

    df = pd.read_csv(metadata)
    train_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["tr"])]
    train_df_x = df["filename"].astype(str).map(lambda x: os.path.join(root, x)).tolist()
    # y==0 is non-blonde and y==1 is blonde
    train_df_y = df["y"].tolist()
    train_df_g = np.array(df["a"].tolist())

    train_dataset = CELEBA(train_df_x, train_df_y, train_df_g)
    val_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["va"])]
    test_df = df[df["split"] == ({"tr": 0, "va": 1, "te": 2}["te"])]

get_celeba()