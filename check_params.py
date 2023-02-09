import os
import json

# Parameters
num_seed = 3
method = 0
dataset = "mcdominoes"
spurious_strength = 0.95
val_size = 1000

file_initial = f"{method}-{dataset}-{spurious_strength}-{val_size}"

# Filenames
files = [i for i in os.listdir() if file_initial in i]
files_without_seed = list(set([i[:-2] for i in files]))

# Results
results = {}

#  For each parameter, store the worst-group accuracy averaged over the seeds
for file_without_seed in files_without_seed:
    best_worst_group_accuracy = 0
    for seed in range(num_seed):
        file_with_seed = file_without_seed + f"-{seed}"
        # print(file_with_seed)
        result_path = os.path.join(file_with_seed, "log.txt")
        with open(result_path) as f:
            lines = f.readlines()
        # print(lines)
        best_worst_group_accuracy += float(lines[-1].split()[4][:-2])
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
