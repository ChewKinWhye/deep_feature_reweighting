#/hpctmp/e0200920/method_2_ablation/2-3-mcdominoes-0.8-1000-1e-3-16-1e-3-8-0/log.txt
#/hpctmp/e0200920/$method-$regularize_mode-$dataset-$spurious_strength-$val_target_size-$weight_decay-$batch_size-$init_lr-$group_size-$seed

import os
import numpy as np
import argparse


base_dir = "/hpctmp/e0200920/method_1_ablation/"

method = "1"
dataset = "mcdominoes"
weight_decay = "1e-3"
batch_size = 16
init_lr = "1e-3"
group_size = 8
num_seed = 3

for spurious_strength in [0.8, 0.9, 0.95, 1]:
    for val_target_size in [1000, 2000, 6000]:

        results = []
        for seed in range(num_seed):
            log_dir = os.path.join(base_dir, f"{method}-{dataset}-{spurious_strength}-{val_target_size}-{weight_decay}-{batch_size}-{init_lr}-{seed}", "log.txt")
            print(log_dir)
            with open(log_dir) as f:
                lines = f.readlines()
            best_validation = float(f"{lines[-1].split()[4][:-2]:.3f}")
            for i in range(len(lines)):
                if float(lines[i].split()[4]) == best_validation:
                    print("Found!")
                    results.append(lines[i-1].split()[4])

        results = np.array(results) * 100
        print(f"{spurious_strength}, {val_target_size}, {np.mean(results):.2f}, {np.std(results):.2f}")
