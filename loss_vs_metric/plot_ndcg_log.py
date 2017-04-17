import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

f = open(sys.argv[1], "r")

#example:
#PLOT Epoch -1 ../data/OSHUMEDQueryLevelNorm/Fold2/train.txt 0.0 ndcg [0.470899470899471, 0.439153439153439, 0.421738817320727, 0.424403591870741, 0.42438884310625, 0.41510415828508, 0.418656649251772, 0.421345857301234, 0.422709106453915, 0.423753745391695] map 0.46835421242 precision [0.650793650793651, 0.619047619047619, 0.603174603174603, 0.599206349206349, 0.587301587301587, 0.555555555555556, 0.555555555555556, 0.555555555555556, 0.555555555555556, 0.541269841269841]


training_process_loss = defaultdict(lambda: dict())
training_process_ndcg = defaultdict(lambda: dict())

for line in f.readlines():
    process = int(re.search(r".condor_out.([0-9]+).", line).group(1))
    run = int(re.search(r"repeated train \[([0-9]+)\]", line).group(1))
    loss = 1 - float(re.search(r"with loss \[([.0-9_]+)\]", line).group(1))
    ndcg = float(re.search(r"train ndcg \[([.0-9_]+)\]", line).group(1))
    training_process_loss[process][run] = loss
    training_process_ndcg[process][run] = ndcg

for subi in range(len(training_process_ndcg.keys())):
    process = list(training_process_ndcg.keys())[subi]
    plt.clf()
    plt.plot(list(training_process_loss[process].keys()), list(training_process_loss[process].values()), 'r--',
             label='1-loss')
    plt.plot(list(training_process_ndcg[process].keys()), list(training_process_ndcg[process].values()), 'g--',
             label='nDCG(training)')
    plt.title(process)
    plt.xlabel("repeated train #")
    plt.ylabel("1-loss or nDCG #")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(sys.argv[1] + "_plots/" + str(process) + ".png")

print(np.mean([len(training_process_loss[process]) for process in training_process_loss]))
