import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re

f = open("random_level_log", "r")

valuedict = defaultdict()

for line in f.readlines():
    fold_num = int(re.search(r"Fold([0-9]+)/", line).group(1))
    random_level = int(re.search(r"trainlog_ListMLE.txt.([0-9]+).", line).group(1))
    iter = int(re.search(r"trainlog_ListMLE.txt.([0-9]+).([0-9]+) ", line).group(2))

    valuedict[(fold_num, random_level, iter)] = (float(line.split()[1]), float(line.split()[2]))

color_list = [np.random.rand(3, ) for i in range(25)]

plt.gcf().subplots_adjust(bottom=0.5)

for fold_num in range(1, 6):

    print "current fold " + str(fold_num)

    avg1_validmax = np.mean([valuedict[(fold_num, 1, i)][0] for i in range(1, 11)])
    avg1_final = np.mean([valuedict[(fold_num, 1, i)][1] for i in range(1, 11)])

    max1_validmax = max([valuedict[(fold_num, 1, i)][0] for i in range(1, 11)])
    max1_final = max([valuedict[(fold_num, 1, i)][1] for i in range(1, 11)])

    min1_validmax = min([valuedict[(fold_num, 1, i)][0] for i in range(1, 11)])
    min1_final = min([valuedict[(fold_num, 1, i)][1] for i in range(1, 11)])

    rand2_validmax = valuedict[(fold_num, 2, 1)][0]
    rand2_final = valuedict[(fold_num, 2, 1)][1]

    train_max_3_validmax = valuedict[(fold_num, 3, 1)][0]
    train_max_3_final = valuedict[(fold_num, 3, 1)][1]

    train_min_3_validmax = valuedict[(fold_num, 3, 2)][0]
    train_min_3_final = valuedict[(fold_num, 3, 2)][1]

    valid_max_3_validmax = valuedict[(fold_num, 4, 1)][0]
    valid_max_3_final = valuedict[(fold_num, 4, 1)][1]

    valid_min_3_validmax = valuedict[(fold_num, 4, 2)][0]
    valid_min_3_final = valuedict[(fold_num, 4, 2)][1]

    test_max_3_validmax = valuedict[(fold_num, 5, 1)][0]
    test_max_3_final = valuedict[(fold_num, 5, 1)][1]

    test_min_3_validmax = valuedict[(fold_num, 5, 2)][0]
    test_min_3_final = valuedict[(fold_num, 5, 2)][1]

    y_value = [avg1_validmax, avg1_final, max1_validmax, max1_final, min1_validmax, min1_final, rand2_validmax,
               rand2_final, train_max_3_validmax, train_max_3_final, train_min_3_validmax, train_min_3_final,
               valid_max_3_validmax, valid_max_3_final, valid_min_3_validmax, valid_min_3_final, test_max_3_validmax,
               test_max_3_final, test_min_3_validmax, test_min_3_final]

    x_pos = range(1,7)+range(8,10)+range(11,13)+range(14,16)+range(17,19)+range(20,22)+range(23,25)+range(26,28)


    x_label = ["avg1_validmax", "avg1_final", "max1_validmax", "max1_final", "min1_validmax", "min1_final","rand2_validmax",
               "rand2_final", "train_max_3_validmax", "train_max_3_final", "train_min_3_validmax", "train_min_3_final",
               "valid_max_3_validmax", "valid_max_3_final", "valid_min_3_validmax", "valid_min_3_final", "test_max_3_validmax",
               "test_max_3_final", "test_min_3_validmax", "test_min_3_final"]

    barlist = plt.bar(x_pos, y_value, align="center", alpha=0.5, width=0.5)
    plt.xticks(x_pos, x_label, rotation='vertical')
    plt.ylabel("MAP")
    plt.title("Fold " + str(fold_num) + " at different random level")
    plt.gca().yaxis.grid(True)
    plt.gca().set_ylim(min(y_value) * 0.9, max(y_value) * 1.1)

    for i in range(len(x_pos)):
        plt.text(x_pos[i] - 0.25, y_value[i], str(y_value[i])[:5],size=5)
        barlist[i].set_color(color_list[i])

    plt.draw()
    plt.savefig("ndcg_log_plots/random_level_" + str(fold_num) + ".pdf")

    plt.clf()

f.close()
