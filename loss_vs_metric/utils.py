import re
import numpy as np
from collections import defaultdict
import sys


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

        print "N (query)= " + str(len(queryDocCount.keys()))
        print "N (documents)= " + str((queryDocCount.values()))
        print "m (features)= " + str(m)
        print queryDocCount

        N = len(queryDocCount.keys())
        n = np.max([len(value) for value in queryDocCount.values()])

        input = np.zeros([N, 1, n, m + 1])
        output = -1 * np.ones([N, n])

        for line in lines:
            score = int(line.split()[0])
            qid = int(re.search(r"qid:([0-9]+).", line).group(1))
            docid = line.strip().split("#docid = ")[1]
            features = [float(tuple.split(":")[1]) for tuple in line.strip().split("#")[0].strip().split()[2:]]
            # print(queryDocCount.keys().index(qid))
            # print(queryDocCount[qid].index(docid))

            input[queryDocCount.keys().index(qid)][0][queryDocCount[qid].index(docid)] = np.asarray(features + [1])
            # print(np.asarray([1] + features))
            output[queryDocCount.keys().index(qid)][queryDocCount[qid].index(docid)] = score

    return input, output, N, n, m
