import sys
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import re

max_vali_map_epoch_index = -1
max_train_map_epoch_index = -1
max_test_map_epoch_index = -1

print_message=sys.argv[1]+" "
for split in ['train', 'vali', 'test']:
    f = open(sys.argv[1], "r")

    # PLOT Epoch -1 ../data/OSHUMEDQueryLevelNorm/Fold2/train.txt 0.0 ndcg [0.470899470899471, 0.439153439153439, 0.421738817320727, 0.424403591870741, 0.42438884310625, 0.41510415828508, 0.418656649251772, 0.421345857301234, 0.422709106453915, 0.423753745391695] map 0.46835421242 precision [0.650793650793651, 0.619047619047619, 0.603174603174603, 0.599206349206349, 0.587301587301587, 0.555555555555556, 0.555555555555556, 0.555555555555556, 0.555555555555556, 0.541269841269841]

    epochs = list()
    training_process_metric = defaultdict(lambda: list())

    for line in f.readlines():
        if split in line and "PLOT " in line:
            epoch = int(re.search(r"Epoch ([\-]*[0-9]+) ", line).group(1))

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

            # print epoch, ndcg1, ndcg5, ndcg10, map, p1, p5, p10
            epochs += [epoch]

            training_process_metric['loss'] += [loss]
            training_process_metric['ndcg1'] += [ndcg1]
            training_process_metric['ndcg5'] += [ndcg5]
            training_process_metric['ndcg10'] += [ndcg10]

            training_process_metric['map'] += [map]

            training_process_metric['p1'] += [p1]
            training_process_metric['p5'] += [p5]
            training_process_metric['p10'] += [p10]
    training_process_metric['loss'][0] = training_process_metric['loss'][1]

    from mpl_toolkits.axes_grid1 import host_subplot
    import mpl_toolkits.axisartist as AA
    import matplotlib.pyplot as plt

    plt.figure(figsize=(20, 10))

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

    host.set_xlim(min(epochs), max(epochs))
    host.set_ylim(min(training_process_metric['loss']), max(training_process_metric['loss']))

    host.set_xlabel("Epoch")
    host.set_ylabel("Loss")
    host.set_title("_".join(re.split('/|\.', sys.argv[1])[4:]) + "_" + split)
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

    p = [None] * 9

    p[1], = host.plot(epochs, training_process_metric['loss'], label="Loss")
    p[2], = par[0].plot(epochs, training_process_metric['map'], label="map")
    p[3], = par[1].plot(epochs, training_process_metric['ndcg1'], label="NDCG@1")
    p[4], = par[2].plot(epochs, training_process_metric['ndcg5'], label="NDCG@5")
    p[5], = par[3].plot(epochs, training_process_metric['ndcg10'], label="NDCG@10")
    p[6], = par[4].plot(epochs, training_process_metric['p1'], label="P@1")
    p[7], = par[5].plot(epochs, training_process_metric['p5'], label="P@5")
    p[8], = par[6].plot(epochs, training_process_metric['p10'], label="P@10")

    host.set_ylim(min(training_process_metric['loss']) - (
        max(training_process_metric['loss']) - min(training_process_metric['loss'])) * 0.1,
                  max(training_process_metric['loss']) + (
                      max(training_process_metric['loss']) - min(training_process_metric['loss'])) * 0.1)
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

    if 'vali' in split:
        # max_vali_map_epoch_index = max(int(0.2 * len(training_process_metric['map'])),
        #                                training_process_metric['map'].index(np.max(training_process_metric['map'])))
        max_vali_map_epoch_index = training_process_metric['map'].index(np.max(training_process_metric['map']))
    elif 'train' in split:
        # max_train_map_epoch_index = max(int(0.2 * len(training_process_metric['map'])),
        #                                 training_process_metric['map'].index(np.max(training_process_metric['map'])))
        max_train_map_epoch_index = training_process_metric['map'].index(np.max(training_process_metric['map']))
    elif 'test' in split:
        # max_test_map_epoch_index = max(int(0.2 * len(training_process_metric['map'])),
        #                                training_process_metric['map'].index(np.max(training_process_metric['map'])))
        max_test_map_epoch_index = training_process_metric['map'].index(np.max(training_process_metric['map']))

    annotation_text_vali = "(Max Validation) at epoch " + str(epochs[max_vali_map_epoch_index]) + " loss " + str(
        training_process_metric["loss"][max_vali_map_epoch_index]) + " map " + str(
        training_process_metric["map"][max_vali_map_epoch_index]) + " ndcg10 " + str(
        training_process_metric["ndcg10"][max_vali_map_epoch_index])

    annotation_text_train = "(Max Train) at epoch " + str(epochs[max_train_map_epoch_index]) + " loss " + str(
        training_process_metric["loss"][max_train_map_epoch_index]) + " map " + str(
        training_process_metric["map"][max_train_map_epoch_index]) + " ndcg10 " + str(
        training_process_metric["ndcg10"][max_train_map_epoch_index])

    annotation_text_test = "(Max Test) at epoch " + str(epochs[max_test_map_epoch_index]) + " loss " + str(
        training_process_metric["loss"][max_test_map_epoch_index]) + " map " + str(
        training_process_metric["map"][max_test_map_epoch_index]) + " ndcg10 " + str(
        training_process_metric["ndcg10"][max_test_map_epoch_index])

    annotation_text_final = "(Final epoch) at epoch " + str(epochs[-1]) + " loss " + str(
        training_process_metric["loss"][-1]) + " map " + str(
        training_process_metric["map"][-1]) + " ndcg10 " + str(
        training_process_metric["ndcg10"][-1])

    host.text(min(epochs) * 1.2, min(training_process_metric['loss']), annotation_text_vali, fontsize=10)
    host.text(min(epochs) * 1.2, min(training_process_metric['loss']) + (
        max(training_process_metric['loss']) - min(training_process_metric['loss'])) * 0.05, annotation_text_train,
              fontsize=10)
    host.text(min(epochs) * 1.2, min(training_process_metric['loss']) + 2 * (
        max(training_process_metric['loss']) - min(training_process_metric['loss'])) * 0.1, annotation_text_test,
              fontsize=10)

    host.text(min(epochs) * 1.2, min(training_process_metric['loss']) + 2 * (
        max(training_process_metric['loss']) - min(training_process_metric['loss'])) * 0.15, annotation_text_final,
              fontsize=10)
    plt.draw()
    plt.savefig("ndcg_log_plots/" + "_".join(re.split('/|\.', sys.argv[1])[4:]) + "_" + split + ".png")

    plt.clf()
    # plt.show()
    f.close()

    # if "vali" in split or "test" in split:
    if "test" in split:
        print_message +=str(training_process_metric["map"][max_vali_map_epoch_index])+" "+str(training_process_metric["map"][-1])+" "

print print_message
