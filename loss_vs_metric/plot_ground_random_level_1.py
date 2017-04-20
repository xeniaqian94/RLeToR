import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re

# f = open(sys.argv[1], "r")
f = open("groundtruth_random_level_1", "r")
folds = defaultdict(lambda: defaultdict(lambda: list()))

for line in f.readlines():
    folds[line.split()[0].split("/")[3]]["valid_MAX"] += [float(line.split()[1])]
    folds[line.split()[0].split("/")[3]]["final_valid"] += [float(line.split()[2])]
    folds[line.split()[0].split("/")[3]]["test_at_valid_MAX"] += [float(line.split()[3])]
    folds[line.split()[0].split("/")[3]]["final_test"] += [float(line.split()[4])]

for fold in folds.keys():
    plt.plot(np.arange(10), folds[fold]["valid_MAX"], label="valid_MAX")
    plt.plot(np.arange(10), folds[fold]["final_valid"], label="final_valid")
    plt.plot(np.arange(10), folds[fold]["test_at_valid_MAX"], label="test_at_valid_MAX")
    plt.plot(np.arange(10), folds[fold]["final_test"], label="final_test")
    plt.title(fold)
    plt.legend()
    plt.xticks(np.arange(10))
    plt.grid()
    plt.draw()
    plt.savefig("ndcg_log_plots/groundtruth_random_level_1_" + fold + ".png")

    plt.clf()
