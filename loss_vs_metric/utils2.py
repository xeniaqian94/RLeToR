import re
import numpy as np
from collections import defaultdict
import sys
from torch.autograd import Variable
import torch

dtype = torch.FloatTensor


def normalize_by_column(matrix):
    max_minus_min = (matrix.max(axis=0) - matrix.min(axis=0))
    max_minus_min[max_minus_min == 0] = 1
    return (matrix - matrix.min(axis=0)) / max_minus_min


def load_data_ListMLE(path, normalized=False):
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

        N = len(queryDocCount.keys())
        n = np.max([len(value) for value in queryDocCount.values()])

        input = np.zeros([N, 1, n, m])
        output = -1 * np.ones([N, n])

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]
            features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
            input[queryDocCount.keys().index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features)
            output[queryDocCount.keys().index(qid)][queryDocCount[qid].index(docid)] = score

    dtype = torch.FloatTensor

    if normalized:
        print "Normalizing, old:"

        for query in queryDocCount.keys():
            # print input[queryDocCount.keys().index(query)][0]

            input[queryDocCount.keys().index(query)][0] = normalize_by_column(
                input[queryDocCount.keys().index(query)][0])
            # print "new "
            # print input[queryDocCount.keys().index(query)][0]
    input_sorted = torch.from_numpy(input).type(dtype)
    output_sorted = torch.from_numpy(output).type(dtype)

    for query in range(output_sorted.size()[0]):
        feature = input_sorted[query][0]
        label = output_sorted[query]

        # print query, label[0:5]

        # order = torch.sort(label, 0, descending=True)[1]  # order=[3,0,2,1]


        order = torch.Tensor(np.argsort(label.numpy())[::-1])
        order = torch.sort(order, 0)[1]  # order=[1,3,2,0] meaning score[0] -> position[1], score[1] -> position[3] ...
        # print feature,label,order

        ordered_feature = dtype(feature.size()[0], feature.size()[1])
        ordered_feature.index_copy_(0, order, feature)  # tensor copy based on the position order
        # print ordered_feature

        ordered_label = dtype(label.size()[0])
        ordered_label.index_copy_(0, order, label)
        # print ordered_label

        input_sorted[query][0] = ordered_feature
        output_sorted[query] = ordered_label

    input_sorted = Variable(input_sorted, requires_grad=False)
    output_sorted = Variable(output_sorted, requires_grad=False)

    input_unsorted = Variable(torch.from_numpy(input).type(dtype), requires_grad=False)
    output_unsorted = Variable(torch.from_numpy(output).type(dtype), requires_grad=False)

    # print input
    # print output
    return input_sorted, output_sorted, input_unsorted, output_unsorted, N, n, m


def load_data_RankSVM(path, normalized=False):
    with open(path, "r") as f:
        print path
        lines = f.readlines()
        queryPosDocCount = defaultdict(lambda: list())
        queryNegDocCount = defaultdict(lambda: list())
        queryDocCount = defaultdict(lambda: list())
        queryOrder = list()
        m = 0  # feature dimension
        for line in lines:
            try:
                score = int(line.split()[0])
                qid = int(re.search(r"qid:([0-9]+).", line).group(1))
                docid = line.strip().split("#docid = ")[1]
                if score > 0:
                    queryPosDocCount[qid] += [docid]
                else:
                    queryNegDocCount[qid] += [docid]
                if qid not in queryOrder:
                    queryOrder += [qid]
                queryDocCount[qid] += [docid]
                features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
                m = np.max([m, len(features)])

            except:
                print("Unexpected error:", sys.exc_info()[0])

        N = max(len(queryPosDocCount.keys()), len(queryNegDocCount.keys()))
        n1 = np.max([len(queryPosDocCount[key]) for key in queryPosDocCount.keys()])
        inputPos = np.zeros([N, 1, n1, m])
        outputPos = -1 * np.ones([N, n1])

        print queryPosDocCount
        print queryNegDocCount

        n2 = np.max([len(queryNegDocCount[key]) for key in queryNegDocCount.keys()])
        inputNeg = np.zeros([N, 1, n2, m])
        outputNeg = -1 * np.ones([N, n2])

        input = np.zeros([N, 1, n1 + n2, m])
        output = -1 * np.ones([N, n1 + n2])

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]
            features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]

            input[queryDocCount.keys().index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features)
            output[queryDocCount.keys().index(qid)][queryDocCount[qid].index(docid)] = score

        print "input before normalizing" + str(input[:2][0][:2])

        if normalized:
            print "Normalizing, old:"
            for query in queryDocCount.keys():
                input[queryDocCount.keys().index(query)][0] = normalize_by_column(
                    input[queryDocCount.keys().index(query)][0])

        print "input after normalizing" + str(input[:2][:2])

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]

            if score > 0:
                inputPos[queryPosDocCount.keys().index(qid)][0][queryPosDocCount[qid].index(docid)] = \
                    input[queryDocCount.keys().index(qid)][0][queryDocCount[qid].index(docid)]
                outputPos[queryPosDocCount.keys().index(qid)][queryPosDocCount[qid].index(docid)] = \
                    output[queryDocCount.keys().index(qid)][queryDocCount[qid].index(docid)]
            else:
                inputNeg[queryNegDocCount.keys().index(qid)][0][queryNegDocCount[qid].index(docid)] = \
                    input[queryDocCount.keys().index(qid)][0][queryDocCount[qid].index(docid)]
                outputNeg[queryNegDocCount.keys().index(qid)][queryNegDocCount[qid].index(docid)] = \
                    output[queryDocCount.keys().index(qid)][queryDocCount[qid].index(docid)]
    return Variable(torch.from_numpy(input).type(dtype), requires_grad=False), Variable(torch.from_numpy(output).type(dtype),
                                                                                        requires_grad=False), Variable(
        torch.from_numpy(inputPos).type(dtype),
        requires_grad=False), Variable(
        torch.from_numpy(inputNeg).type(dtype), requires_grad=False), Variable(torch.from_numpy(outputPos).type(dtype),
                                                                   requires_grad=False), Variable(
        torch.from_numpy(outputNeg).type(dtype),
        requires_grad=False), N, n1, n2, m
