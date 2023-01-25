import os
import json

# Parameters
method = "COND-CORAL"

# Filenames
files = [i for i in os.listdir("/hpctmp/e0200920") if method in i]

# Results
results = {}

#  For each parameter, store the worst-group accuracy averaged over the seeds
for file_name in files:
    with open(os.path.join("/hpctmp/e0200920", file_name, "log.txt")) as f:
        lines = f.readlines()
    try:
        best_worst_group_accuracy = float(lines[-1].split()[4][:-2])
        results[file_name] = best_worst_group_accuracy
    except:
        print(file_name)
# print(results)
# Print out the best hyperparameter and the worst group accuracy
max_value = 0
best_param = None
for key, value in results.items():
    if value > max_value:
        max_value = value
        best_param = key
print(best_param, max_value)
