import re
import numpy as np
from collections import defaultdict
import sys
from torch.autograd import Variable
import torch
import time

dtype = torch.FloatTensor


def normalize_by_column(matrix):
    max_minus_min = (matrix.max(axis=0) - matrix.min(axis=0))
    max_minus_min[max_minus_min == 0] = 1
    return (matrix - matrix.min(axis=0)) / max_minus_min


def load_data_ListMLE(path, normalized=False):
    with open(path, "r") as f:
        lines = f.readlines()
        queryOrder = list()
        queryDocCount = defaultdict(lambda: list())
        m = 0  # feature dimension
        for line in lines:
            try:
                qid = int(re.search(r"qid:([0-9]+).", line).group(1))
                docid = line.strip().split("#docid = ")[1]
                if qid not in queryOrder:
                    queryOrder += [qid]
                queryDocCount[qid] += [docid]
                features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
                m = np.max([m, len(features)])

            except:
                print("Unexpected error:", sys.exc_info()[0])

        N = len(queryOrder)
        n = np.max([len(value) for value in queryDocCount.values()])

        input = np.zeros([N, 1, n, m])
        output = -1 * np.ones([N, n])

        print queryOrder

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]
            features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
            input[queryOrder.index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features)
            output[queryOrder.index(qid)][queryDocCount[qid].index(docid)] = score

        dtype = torch.FloatTensor

        print path
        print "input unnormalized " + str(input)
        print "output unnormalized " + str(output)

        if normalized:
            print "Normalizing, old:"

            for query in queryOrder:
                input[queryOrder.index(query)][0] = normalize_by_column(
                    input[queryOrder.index(query)][0])

        input_sorted = torch.from_numpy(input).type(dtype)
        output_sorted = torch.from_numpy(output).type(dtype)

        for query in range(output_sorted.size()[0]):
            np.random.seed(int(time.time()))
            feature = input_sorted[query][0]
            label = output_sorted[query]

            order_numpy=np.argsort(label.numpy())[::-1]


            # Random level = 1 select a groundtruth for each query, fix as input

            sorted_label=[label[int(i)] for i in order_numpy]

            valid_doc_num = np.sum(np.asarray(sorted_label)>-1)

            upper_limit_n = np.sum(np.asarray(sorted_label)>0)


            if upper_limit_n > 0: # Has at least one positive case
                until_grade_2 = np.sum(np.asarray(sorted_label)>1)
                order_numpy = list(np.random.permutation(order_numpy[:until_grade_2])) + list(
                    np.random.permutation(order_numpy[until_grade_2:upper_limit_n])) + list(order_numpy[upper_limit_n:])

            # Random level = 1 finished

            order=torch.Tensor(order_numpy)


            order = torch.sort(order, 0)[
                1]  # order=[1,3,2,0] meaning score[0] -> position[1], score[1] -> position[3] ...
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

        print "input sorted " + str(input_sorted)
        print "output sorted " + str(output_sorted)
        print "input unsorted " + str(input_unsorted)
        print "output unsorted " + str(output_unsorted)

        return input_sorted, output_sorted, input_unsorted, output_unsorted, N, n, m


def load_data_RankSVM(path, normalized=False):
    with open(path, "r") as f:
        print path
        lines = f.readlines()
        queryPosDocCount = defaultdict(lambda: list())
        queryNegDocCount = defaultdict(lambda: list())
        queryOrder = list()
        queryDocCount = defaultdict(lambda: list())
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

        N = len(queryOrder)
        n1 = np.max([len(queryPosDocCount[key]) for key in queryPosDocCount.keys()])
        inputPos = np.zeros([N, 1, n1, m])
        outputPos = -1 * np.ones([N, n1])

        print queryPosDocCount
        print queryNegDocCount
        print queryDocCount
        print queryOrder

        n2 = np.max([len(queryNegDocCount[key]) for key in queryNegDocCount.keys()])
        inputNeg = np.zeros([N, 1, n2, m])
        outputNeg = -1 * np.ones([N, n2])

        n = np.max([len(queryDocCount[key]) for key in queryDocCount.keys()])

        input = np.zeros([N, 1, n, m])
        output = -1 * np.ones([N, n])

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]
            features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]

            input[queryOrder.index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features)
            output[queryOrder.index(qid)][queryDocCount[qid].index(docid)] = score

        print "input before normalizing" + str(input)

        if normalized:
            print "Normalizing, old:"
            for query in queryOrder:
                input[queryOrder.index(query)][0] = normalize_by_column(
                    input[queryOrder.index(query)][0])

        print "input after normalizing" + str(input)

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]

            if score > 0:
                inputPos[queryOrder.index(qid)][0][queryPosDocCount[qid].index(docid)] = \
                    input[queryOrder.index(qid)][0][queryDocCount[qid].index(docid)]
                outputPos[queryOrder.index(qid)][queryPosDocCount[qid].index(docid)] = \
                    output[queryOrder.index(qid)][queryDocCount[qid].index(docid)]
            else:
                inputNeg[queryOrder.index(qid)][0][queryNegDocCount[qid].index(docid)] = \
                    input[queryOrder.index(qid)][0][queryDocCount[qid].index(docid)]
                outputNeg[queryOrder.index(qid)][queryNegDocCount[qid].index(docid)] = \
                    output[queryOrder.index(qid)][queryDocCount[qid].index(docid)]
    return Variable(torch.from_numpy(input).type(dtype), requires_grad=False), Variable(
        torch.from_numpy(output).type(dtype),
        requires_grad=False), Variable(
        torch.from_numpy(inputPos).type(dtype),
        requires_grad=False), Variable(
        torch.from_numpy(inputNeg).type(dtype), requires_grad=False), Variable(torch.from_numpy(outputPos).type(dtype),
                                                                               requires_grad=False), Variable(
        torch.from_numpy(outputNeg).type(dtype),
        requires_grad=False), N, n1, n2, m
