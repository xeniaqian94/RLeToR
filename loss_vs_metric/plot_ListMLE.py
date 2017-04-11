import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

f = open(sys.argv[1], "r")

epochs = list()
training_process_metric = defaultdict(lambda: list())

for line in f.readlines():
    if "PLOT" in line:
        epoch = int(re.search(r"Epoch ([0-9]+) ", line).group(1))

        loss = float(re.search(r" ([0-9.]+) ndcg", line).group(1))

        ndcg = re.search(r"ndcg \[([0-9., ]+)\]", line).group(1).split(", ")
        ndcg1 = float(ndcg[0])
        ndcg5 = float(ndcg[4])
        ndcg10 = float(ndcg[9])

        map = float(re.search(r"map ([0-9.]+) precision", line).group(1))

        precision = re.search(r"precision \[([0-9., ]+)\]", line).group(1).split(", ")
        p1 = float(precision[0])
        p5 = float(precision[4])
        p10 = float(precision[9])

        print epoch, ndcg1, ndcg5, ndcg10, map, p1, p5, p10
        epochs += [epoch]

        training_process_metric['loss'] += [loss]
        training_process_metric['ndcg1'] += [ndcg1]
        training_process_metric['ndcg5'] += [ndcg5]
        training_process_metric['ndcg10'] += [ndcg10]

        training_process_metric['map'] += [map]

        training_process_metric['p1'] += [p1]
        training_process_metric['p5'] += [p5]
        training_process_metric['p10'] += [p10]

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))

host = host_subplot(111, axes_class=AA.Axes)
plt.subplots_adjust(right=0.6)

par = list()
offset = list()
new_fixed_axis = list()

for i in range(7):
    par += [host.twinx()]
    offset += [35 * (i + 1)]
    new_fixed_axis += [par[i].get_grid_helper().new_fixed_axis]
    par[i].axis["right"] = new_fixed_axis[i](loc="right",
                                             axes=par[i],
                                             offset=(offset[i], 0))
    par[i].axis["right"].toggle(all=True)

#
# par1 = host.twinx()
# par2 = host.twinx()
# #par3 = host.twinx()
#
#
#
#
# offset = 60
# new_fixed_axis = par2.get_grid_helper().new_fixed_axis
#
# par2.axis["right"] = new_fixed_axis(loc="right",
#                                     axes=par2,
#                                     offset=(offset, 0))
#
# par2.axis["right"].toggle(all=True)

# plt.figure(figsize=(3.841, 7.195), dpi=100)

host.set_xlim(min(epochs), max(epochs))
host.set_ylim(min(training_process_metric['loss']), max(training_process_metric['loss']))

host.set_xlabel("Epoch")
host.set_ylabel("Loss")
host.set_title(sys.argv[1])
host.grid()
# par1.set_ylabel("MAP")
# par2.set_ylabel("NDCG@1")
par[0].set_ylabel("MAP")
par[1].set_ylabel("NDCG@1")
par[2].set_ylabel("NDCG@5")
par[3].set_ylabel("NDCG@10")
par[4].set_ylabel("P@1")
par[5].set_ylabel("P@5")
par[6].set_ylabel("P@10")

p = [None]*9


p[1], = host.plot(epochs, training_process_metric['loss'], label="Loss")
p[2], = par[0].plot(epochs, training_process_metric['map'], label="map")
p[3], = par[1].plot(epochs, training_process_metric['ndcg1'], label="NDCG@1")
p[4], = par[2].plot(epochs, training_process_metric['ndcg5'], label="NDCG@5")
p[5], = par[3].plot(epochs, training_process_metric['ndcg10'], label="NDCG@10")
p[6], = par[4].plot(epochs, training_process_metric['p1'], label="P@1")
p[7], = par[5].plot(epochs, training_process_metric['p5'], label="P@5")
p[8], = par[6].plot(epochs, training_process_metric['p10'], label="P@10")

par[0].set_ylim(min(training_process_metric['map']) * 0.9, max(training_process_metric['map']) * 1.1)
par[1].set_ylim(min(training_process_metric['ndcg1']) * 0.9, max(training_process_metric['ndcg1']) * 1.1)
par[2].set_ylim(min(training_process_metric['ndcg5']) * 0.9, max(training_process_metric['ndcg5']) * 1.1)
par[3].set_ylim(min(training_process_metric['ndcg10']) * 0.9, max(training_process_metric['ndcg10']) * 1.1)
par[4].set_ylim(min(training_process_metric['p1']) * 0.9, max(training_process_metric['p1']) * 1.1)
par[5].set_ylim(min(training_process_metric['p5']) * 0.9, max(training_process_metric['p5']) * 1.1)
par[6].set_ylim(min(training_process_metric['p10']) * 0.9, max(training_process_metric['p10']) * 1.1)

host.legend()

host.axis["left"].label.set_color(p[1].get_color())
for i in range(7):
    par[i].axis["right"].label.set_color(p[i + 2].get_color())

plt.draw()
plt.savefig("ndcg_log_plots/" + sys.argv[1] + ".png")
# plt.show()


# for subi in range(len(training_process_ndcg.keys())):
#     process = list(training_process_ndcg.keys())[subi]
#     plt.clf()
#     plt.plot(list(training_process_loss[process].keys()), list(training_process_loss[process].values()), 'r--',
#              label='1-loss')
#     plt.plot(list(training_process_ndcg[process].keys()), list(training_process_ndcg[process].values()), 'g--',
#              label='nDCG(training)')
#     plt.title(process)
#     plt.xlabel("repeated train #")
#     plt.ylabel("1-loss or nDCG #")
#     plt.legend(loc='upper left')
#     plt.grid(True)
#     plt.savefig(sys.argv[1] + "_plots/" + str(process) + ".png")
#
# print(np.mean([len(training_process_loss[process]) for process in training_process_loss]))
