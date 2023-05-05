import os
import numpy as np
import argparse


base_dir = "/hpctmp/e0200920/method_2-0_ablation/"
method = "2-0"
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
            if method == 0 or method == 1 or method == "DFR":
                log_dir = os.path.join(base_dir, f"{method}-{dataset}-{spurious_strength}-{val_target_size}-{weight_decay}-{batch_size}-{init_lr}-{seed}", "log.txt")
            else:
                log_dir = os.path.join(base_dir, f"{method}-{dataset}-{spurious_strength}-{val_target_size}-{weight_decay}-{batch_size}-{init_lr}-{group_size}-{seed}", "log.txt")
            try:
                with open(log_dir) as f:
                    lines = f.readlines()
                if method == "DFR":
                    best_test = round(float(lines[-1].split()[3][:-2]), 3)
                else:
                    best_validation = round(float(lines[-1].split()[4][:-2]), 3)
            except:
                print(f"Cannot find: {log_dir}")
                continue
            if method == "DFR":
                results.append(best_test)
            else:
                for i in range(len(lines)):
                    try:
                        if lines[i].split()[1] == "Validation" and float(lines[i].split()[3]) == best_validation:
                            results.append(float(lines[i-1].split()[3]))
                            break
                    except:
                        continue
        if len(results) != num_seed:
            print("MISSING RESULTS")
        results = np.array(results) * 100
        print(f"{spurious_strength}, {val_target_size}, {np.mean(results):.2f}, {np.std(results):.2f}")
