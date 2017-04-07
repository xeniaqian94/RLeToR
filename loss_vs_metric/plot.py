import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt

f=open(sys.argv[1],"r")

training_process_loss=defaultdict(lambda:dict())
training_process_ndcg=defaultdict(lambda:dict())

for line in f.readlines():
    process=int(re.search(r".condor_out.([0-9]+).", line).group(1))
    run=int(re.search(r"repeated train \[([0-9]+)\]", line).group(1))
    loss=float(re.search(r"with loss \[([.0-9_]+)\]", line).group(1))
    ndcg=float(re.search(r"train ndcg \[([.0-9_]+)\]", line).group(1))
    training_process_loss[process][run]=loss
    training_process_ndcg[process][run]=ndcg

# f, axarr = plt.subplots(len(training_process_ndcg.keys()), sharex=True,figsize=(8, 40))
#
#
# for subi in range(len(training_process_ndcg.keys())):
#     process=list(training_process_ndcg.keys())[subi]
#
#     axarr[subi].plot(list(training_process_loss[process].keys()),list(training_process_loss[process].values()),'r--',list(training_process_ndcg[process].keys()),list(training_process_ndcg[process].values()),'g.')
#     axarr[subi].set_title(process)
#     axarr[subi].grid(True)
#
#
# f.savefig('full_figure.png')

for subi in range(len(training_process_ndcg.keys())):
    process = list(training_process_ndcg.keys())[subi]
    plt.clf()
    plt.plot(list(training_process_loss[process].keys()),list(training_process_loss[process].values()),'r--',label='loss')
    plt.plot(list(training_process_ndcg[process].keys()),list(training_process_ndcg[process].values()),'g--',label='nDCG(training)')
    plt.title(process)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.savefig(sys.argv[1]+"_plots/"+str(process)+".png")
