import re
import numpy as np
from collections import defaultdict
import sys
from torch.autograd import Variable
import torch


def load_data(path):
    with open(path, "r") as f:
        lines = f.readlines()
        queryDocCount = defaultdict(lambda: list())
        m = 0  # feature dimension
        for line in lines:
            try:
                score = int(line.split()[0])
                qid = int(re.search(r"qid:([0-9]+).", line).group(1))
                docid = line.strip().split("#docid = ")[1]
                queryDocCount[qid] += [docid]
                features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
                m = np.max([m, len(features)])

            except:
                print("Unexpected error:", sys.exc_info()[0])

        # print "N (query)= " + str(len(queryDocCount.keys()))
        # print "N (documents)= " + str((queryDocCount.values()))
        # print "m (features)= " + str(m)
        # print queryDocCount

        N = len(queryDocCount.keys())
        n = np.max([len(value) for value in queryDocCount.values()])

        input = np.zeros([N, 1, n, m])
        output = -1 * np.ones([N, n])

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]
            features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
            # print(queryDocCount.keys().index(qid))
            # print(queryDocCount[qid].index(docid))

            input[queryDocCount.keys().index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features)
            # print(np.asarray([1] + features))
            output[queryDocCount.keys().index(qid)][queryDocCount[qid].index(docid)] = score

    dtype = torch.FloatTensor
    input=torch.from_numpy(input).type(dtype)
    output=torch.from_numpy(output).type(dtype)

    for query in range(output.size()[0]):

        feature=input[query][0]
        label = output[query]

        order = torch.sort(label, 0, descending=True)[1]  # order=[3,0,2,1]
        order = torch.sort(order, 0)[
            1]  # order=[1,3,2,0] meaning score[0] -> position[1], score[1] -> position[3] ...
        # print feature,label,order

        ordered_feature = dtype(feature.size()[0],feature.size()[1])
        ordered_feature.index_copy_(0, order, feature)  # tensor copy based on the position order
        # print ordered_feature

        ordered_label = dtype(label.size()[0])
        ordered_label.index_copy_(0, order, label)
        # print ordered_label

        input[query][0]=ordered_feature
        output[query]=ordered_label

    input = Variable(input, requires_grad=False)
    output = Variable(output, requires_grad=False)

    # print input
    # print output
    return input, output, N, n, m
