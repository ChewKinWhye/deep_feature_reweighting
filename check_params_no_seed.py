import os
import json

# Parameters
method = "MTL"

# Filenames
files = [i for i in os.listdir() if method in i]

# Results
results = {}

for result_file in files:
    result_path = os.path.join(result_file, "log.txt")
    with open(result_path) as f:
        lines = f.readlines()
    # print(lines)
    try:
        best_worst_group_accuracy = float(lines[-1].split()[4][:-2])
        results[result_file] = best_worst_group_accuracy
    except:
        continue

# print(results)
# Print out the best hyperparameter and the worst group accuracy
max_value = 0
best_param = None
for key, value in results.items():
    if value > max_value:
        max_value = value
        best_param = key
print(best_param, max_value)
