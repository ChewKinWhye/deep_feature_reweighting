import os
import json

# Parameters
num_seed = 3
method = "3-1"
dataset = "mcdominoes"
spurious_strength = 1
val_size = 1000

base_path = f"/hpctmp/e0200920/method_{method}"
file_initial = f"{method}-{dataset}-{spurious_strength}-{val_size}"

# Filenames
files = [i for i in os.listdir(base_path) if file_initial in i]
files_without_seed = list(set([i[:-2] for i in files]))

# Results
results = {}

#  For each parameter, store the worst-group accuracy averaged over the seeds
for file_without_seed in files_without_seed:
    best_worst_group_accuracy = 0
    for seed in range(num_seed):
        file_with_seed = file_without_seed + f"-{seed}"
        # print(file_with_seed)
        result_path = os.path.join(base_path, file_with_seed, "log.txt")
        if not os.path.exists(result_path):
            print(result_path)
            continue
        with open(result_path) as f:
            lines = f.readlines()
        # print(lines)
        try:
            best_worst_group_accuracy += float(lines[-1].split()[4][:-2])
        except:
            print(result_path)
            best_worst_group_accuracy = 0
    results[file_without_seed] = best_worst_group_accuracy / num_seed

# print(results)
# Print out the best hyperparameter and the worst group accuracy
max_value = 0
best_param = None
for key, value in results.items():
    if value > max_value:
        max_value = value
        best_param = key
print(best_param, max_value)
