from collections import defaultdict

import re
import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import utils
import numpy as np
import os
import time


def get_MAP(split, iter):
    filename = "/".join(args.model_path.split("/")[:-1]) + "/trainlog_ListMLE.txt.1." + str(iter)
    f = open(filename, "r")

    # PLOT Epoch -1 ../data/OSHUMEDQueryLevelNorm/Fold2/train.txt 0.0 ndcg [0.470899470899471, 0.439153439153439, 0.421738817320727, 0.424403591870741, 0.42438884310625, 0.41510415828508, 0.418656649251772, 0.421345857301234, 0.422709106453915, 0.423753745391695] map 0.46835421242 precision [0.650793650793651, 0.619047619047619, 0.603174603174603, 0.599206349206349, 0.587301587301587, 0.555555555555556, 0.555555555555556, 0.555555555555556, 0.555555555555556, 0.541269841269841]
    epochs = list()
    training_process_metric = defaultdict(lambda: list())

    for line in f.readlines():
        if split in line and "PLOT " in line:
            epoch = int(re.search(r"Epoch ([\-]*[0-9]+) ", line).group(1))
            map = float(re.search(r"map ([0-9.]+) precision", line).group(1))
            epochs += [epoch]
            training_process_metric['map'] += [map]
    f.close()

    return training_process_metric['map'][-1]

def get_index(split,method):
    map_10=[]
    for iter in range(1,11):
        map_10+=[get_MAP(split,iter)]
    map_10=np.array(map_10)

    if method=="MAX":
        return map_10.argsort()[-3:][::-1]+1
    elif method=="MIN":
        return map_10.argsort()[:3]+1

dtype = torch.FloatTensor

# pytorch example referece: https://github.com/jcjohnson/pytorch-examples/blob/master/autograd/two_layer_net_autograd.py
# regression/main.py, mnist/main.py

# nDCG calculation follows perl script Eval-Score-Score 3.0 from LeToR benchmark dataset, however, the formula was implemented
# slightly buggy from line 252 - 255. Position DCG was from index 1 instead of 0.

parser = argparse.ArgumentParser(description='PyTorch ListMLE Example')

# TD2003/Fold1
parser.add_argument('--training_set', type=str, default="../data/TD2003/Fold1/trainingset.txt",
                    help='training set')
parser.add_argument('--valid_set', type=str, default="../data/TD2003/Fold1/validationset.txt",
                    help='validation set')
parser.add_argument('--test_set', type=str, default="../data/TD2003/Fold1/testset.txt",
                    help='test set')

parser.add_argument('--test_output', type=str, default="../data/TD2003/Fold1/testoutput.txt",
                    help='test output')
parser.add_argument('--train_output', type=str, default="../data/TD2003/Fold1/trainoutput.txt",
                    help='train output')
parser.add_argument('--valid_output', type=str, default="../data/TD2003/Fold1/validoutput.txt",
                    help='valid output')

parser.add_argument('--random_level', type=int, default=1,
                    help="random level 1 = randomly select one ground-truth throughout training, random level 2 = dynamically & randomly select ground-truth every epoch ")
parser.add_argument('--iter', type=int, default=1,
                    help="iteration")
parser.add_argument('--model_path', type=str, default="../data/TD2003/Fold1/model.txt",
                    help='model path')
parser.add_argument('--load_model_path', type=str, default="../data/TD2003/Fold1/model.txt",
                    help='model path')

parser.add_argument('--eval_output', type=str, default="../data/TD2003/Fold1/evaloutput.txt",
                    help='eval output path')
parser.add_argument('--list_cutoff', type=int, default=100, metavar='list_cutoff',
                    help='result list cutoff')

# OSHUMED-Normed
# parser.add_argument('--training_set', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/train.txt",
#                     help='training set')
# parser.add_argument('--valid_set', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/vali.txt",
#                     help='validation set')
# parser.add_argument('--test_set', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/test.txt",
#                     help='test set')
# parser.add_argument('--test_output', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/testoutput.txt",
#                     help='test output')
# parser.add_argument('--train_output', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/trainoutput.txt",
#                     help='train output')
# parser.add_argument('--valid_output', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/validoutput.txt",
#                     help='valid output')
# parser.add_argument('--model_path', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/model.txt",
#                     help='model path')
# parser.add_argument('--eval_output', type=str, default="../data/OSHUMEDQueryLevelNorm/Fold1/evaloutput.txt",
#                     help='eval output path')

# OSHUMED-unnormalized
# parser.add_argument('--training-set', type=str, default="../data/Feature-min/Fold1/trainingset.txt",
#                     help='training set')
# # parser.add_argument('--validation-set', type=str, default="../data/TD2003/Fold1/validationset.txt",
# #                     help='validation set')
# parser.add_argument('--test-set', type=str, default="../data/Feature-min/Fold1/testset.txt",
#                     help='test set')
# parser.add_argument('--test_output', type=str, default="../data/Feature-min/Fold1/testoutput.txt",
#                     help='test output')
# parser.add_argument('--train_output', type=str, default="../data/Feature-min/Fold1/trainoutput.txt",
#                     help='train output')
# parser.add_argument('--model_path', type=str, default="../data/Feature-min/Fold1/model.txt",
#                     help='model path')
# parser.add_argument('--eval_output', type=str, default="../data/Feature-min/Fold1/evaloutput.txt",
#                     help='eval output path')

# toy example
# parser.add_argument('--training_set', type=str, default="../data/toy/train.dat",
#                     help='training set')
# parser.add_argument('--valid_set', type=str, default="../data/toy/test.dat",
#                     help='validation set')
# parser.add_argument('--test_set', type=str, default="../data/toy/test.dat",
#                     help='test set')
# parser.add_argument('--test_output', type=str, default="../data/toy/test_output.txt",
#                     help='test output')
# parser.add_argument('--train_output', type=str, default="../data/toy/train_output.txt",
#                     help='train output')
# parser.add_argument('--valid_output', type=str, default="../data/toy/test_output.txt",
#                     help='valid output')
# parser.add_argument('--model_path', type=str, default="../data/toy/model.dat",
#                     help='model path')
# parser.add_argument('--eval_output', type=str, default="../data/toy/evaloutput.txt",
#                     help='eval output path')

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load_model', type=bool, default=True, metavar='S',
                    help='whether to load pre-trained model?')
parser.add_argument('--query_dimension_normalization', type=bool, default=True, metavar='S',
                    help='whether to normalize by query-dimension?')

args = parser.parse_args()

input_string = defaultdict(lambda: list())

torch.manual_seed(args.seed)

args.test_output = args.test_output + "." + str(args.random_level) + "." + str(args.iter)
args.train_output = args.train_output + "." + str(args.random_level) + "." + str(args.iter)
args.valid_output = args.valid_output + "." + str(args.random_level) + "." + str(args.iter)

args.model_path = args.model_path + "." + str(args.random_level) + "." + str(args.iter)
args.eval_output = args.eval_output + "." + str(args.random_level) + "." + str(args.iter)

input_sorted, output_sorted, input_unsorted, output_unsorted, N, n, m = utils.load_data_ListMLE(
    args.training_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

if args.random_level == 1:
    torch.save(input_sorted, args.model_path.replace("model", "input"))
    torch.save(output_sorted, args.model_path.replace("model", "output"))
elif args.random_level>1:
    if args.random_level == 2:
        index_list=range(1,11)
    elif args.random_level == 3:
        if args.iter==1:
            index_list=get_index("train","MAX")
        else:
            index_list=get_index("train","MIN")
        print "Read from evallog select argmax 3"

    elif args.random_level == 4:
        if args.iter==1:
            index_list=get_index("vali","MAX")
        else:
            index_list=get_index("vali","MIN")
        print "Read from evallog select "

    elif args.random_level == 5:
        if args.iter==1:
            index_list=get_index("test","MAX")
        else:
            index_list=get_index("test","MIN")
        print "Read from evallog select "

    print index_list
    input_sorted = torch.load("/".join(args.model_path.split("/")[:-1]) + "/input.txt.1." + str(index_list[0]))
    output_sorted = torch.load("/".join(args.model_path.split("/")[:-1]) + "/output.txt.1." + str(index_list[0]))
    for iter in index_list[1:]:
        input_sorted = torch.cat(
            [input_sorted, torch.load("/".join(args.model_path.split("/")[:-1]) + "/input.txt.1." + str(iter))])
        output_sorted = torch.cat(
            [output_sorted, torch.load("/".join(args.model_path.split("/")[:-1]) + "/output.txt.1." + str(iter))])

print "input sorted size " + str(input_sorted.size())
print "output sorted size " + str(output_sorted.size())

input_test_sorted, output_test_sorted, input_test_unsorted, output_test_unsorted, N_test, n_test, m_test = utils.load_data_ListMLE(
    args.test_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

input_valid_sorted, output_valid_sorted, input_valid_unsorted, output_valid_unsorted, N_valid, n_valid, m_valid = utils.load_data_ListMLE(
    args.valid_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

print "N " + str(N)
print "n " + str(n)
print "m " + str(m)


class Net(nn.Module):
    def __init__(self, m):  # m is the feature dimension except x0 term
        super(Net, self).__init__()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, m), stride=(1, m),
                               bias=True)  # implicitly contains a learnable bias

        self.conv2.weight.data.zero_()
        # self.conv2.bias.data.zero_()

        if args.load_model and os.path.exists(args.load_model_path):
            self.load_state_dict(torch.load(args.load_model_path))
            print "Pre-trained model loaded"

            # self.tanh = nn.Tanh()
            # self.conv2_prev_weight = self.conv2.weight.data.clone()
            # self.conv2_prev_bias = self.conv2.bias.data.clone()

    def forward(self, input):
        # return self.tanh(self.conv2(input))
        return self.conv2(input)

    def seqMLELoss(self, scores, output):
        neg_log_sum = Variable(torch.zeros(1))
        for query in range(scores.size()[0]):

            np.random.seed(int(time.time()))

            score = scores[query].squeeze()  # e.g. score ordered by groundtruth label
            label = output[query].squeeze()  # e.g. label ordered by groundtruth, not existing doc as -1
            valid_doc_num = torch.sum((label > -1).data)

            # print valid_doc_num
            exp_g = score.exp()

            upper_limit_n = torch.sum((label > 0).data)

            if upper_limit_n > 0:

                P_list = Variable(dtype(upper_limit_n))

                # if args.random_level == 2:
                #
                #     total_sum = torch.sum(exp_g[:valid_doc_num])
                #     denom = Variable(dtype(upper_limit_n))
                #     denom[0] = total_sum
                #
                #     until_grade_2 = torch.sum((label > 1).data)
                #
                #     perm = list(np.random.permutation(np.arange(until_grade_2))) + list(
                #         np.random.permutation(np.arange(until_grade_2, upper_limit_n)))
                #
                #     for i in range(1, upper_limit_n):
                #         denom[i] = denom[i - 1] - exp_g[perm[i - 1]]
                #     for i in range(upper_limit_n):
                #         P_list[i] = exp_g[perm[i]] / denom[i]

                # if args.random_level == 1 or args.random_level == 21:
                for i in range(upper_limit_n):
                    P_list[i] = exp_g[i] / torch.sum(exp_g[i:valid_doc_num])  # one-hot groundtruth

                neg_log_sum -= sum(P_list.log())  # equation 9 in ListMLE paper
        # print neg_log_sum

        return neg_log_sum

    def print_param(self):
        print [param.data for param in self.conv2.parameters()]

    def save_scores(self, scores_test, output_test, test_output_path):  # the feature list was unordered

        if os.path.exists(test_output_path):
            os.remove(test_output_path)

        if scores_test.size()[0] == 0:
            scores_test.unsqueeze(0)

        for query in range(scores_test.size()[0]):
            label_test = output_test[query].squeeze()
            valid_doc_num_test = torch.sum((label_test > -1).data)
            scores_write = scores_test.data[query].squeeze().numpy()[:valid_doc_num_test]
            np.savetxt(open(test_output_path, "a"), scores_write)
            # if query==0:
            #     print "Max document for query 1 "+str(np.argmax(scores_write))

    def eval(self, input_this, output_this, outputpath_this, trainpath_this, evalpath_this):
        scores_this = model.forward(input_this)

        model.save_scores(scores_this, output_this, outputpath_this)

        os.system(
            "perl Eval-Score-3.0.pl " + trainpath_this + " " + outputpath_this + " " + evalpath_this + " 0")

        with open(evalpath_this, "r") as eval_f:
            for line in eval_f.readlines():
                if ("precision:	") in line:
                    precision = [float(value) for value in line.split()[1:11]]
                elif ("MAP:	") in line:
                    map = float(line.split()[1])
                elif ("NDCG:	") in line:
                    ndcg = [float(value) for value in line.split()[1:11]]

        print "PLOT Epoch " + str(epoch) + " " + trainpath_this + " " + str(neg_log_sum_loss.data[0]) + " ndcg " + str(
            ndcg) + " map " + str(map) + " precision " + str(precision)


model = Net(m)

print args
# prev_loss = [float('inf') for i in range(input_sorted.data.size()[0])]
prev_loss = float('inf')
original_lr = args.lr

optimizer = torch.optim.Adam(model.parameters(), lr=original_lr)

lr_scaled = False

ndcg = np.zeros(10)
precision = np.zeros(10)

epoch = -1

neg_log_sum_loss = Variable(torch.zeros(1))

model.eval(input_unsorted, output_unsorted, args.train_output, args.training_set, args.eval_output + ".train")
model.eval(input_valid_unsorted, output_valid_unsorted, args.valid_output, args.valid_set,
           args.eval_output + ".valid")
model.eval(input_test_unsorted, output_test_unsorted, args.test_output, args.test_set,
           args.eval_output + ".test")

fold_num = int(re.search(r"Fold([0-9]+)/", args.model_path).group(1))

for epoch in range(args.epochs):

    # if args.random_level == 21:
    #     if epoch % (args.epochs / len(input_string[(fold_num, args.iter)])) == 0:
    #         selected_index = int(epoch / (args.epochs / len(input_string[(fold_num, args.iter)])))
    #         print fold_num, args.iter, selected_index,input_string[(fold_num, args.iter)][min(selected_index,2)]
    #
    #         input_sorted = torch.load(
    #             "/".join(args.model_path.split("/")[:-1]) + "/input.txt.1." + str(
    #                 input_string[(fold_num, args.iter)][min(selected_index,2)]))

    # for query in range(input_sorted.data.size()[0]):
    # for k in range(10):

    # Forward pass
    # model.print_param()
    #     scores = model.forward(input_sorted[query].unsqueeze(0))  # N * n tensor

    scores = model.forward(input_sorted)  # N * n tensor
    # print "score " + str(scores)
    # print output

    # Reset gradients
    # model.reset_grad()
    optimizer.zero_grad()

    # Backward pass
    # neg_log_sum_loss = model.seqMLELoss(scores, output_sorted[query].unsqueeze(0))
    neg_log_sum_loss = model.seqMLELoss(scores, output_sorted)
    # print neg_log_sum_loss

    neg_log_sum_loss.backward()

    # print "Before gradient"
    # Apply gradients
    # for param in model.conv2.parameters():
    #     param.data.add_(-args.lr * param.grad.data)

    optimizer.step()

    # print "After gradient"
    # model.print_param()

    # Stop criterion

    if neg_log_sum_loss.data[0] < prev_loss:
        # if neg_log_sum_loss.data[0] < prev_loss[query]:
        # model.print_param()
        # Save the model see discussion: https: // discuss.pytorch.org / t / saving - torch - models / 838 / 4

        if abs(neg_log_sum_loss.data[0] - prev_loss) < 1e-3:
            # if abs(neg_log_sum_loss.data[0] - prev_loss[query]) < 1e-5:
            torch.save(model.state_dict(), open(args.model_path, "w"))
            # scores_test = model.forward(input_test)
            # model.save_scores(scores_test, output_test, args.test_output)
            break
        if (epoch % 20 == 0):
            # print model.print_param()
            # print "input unsorted "+str(input_unsorted)
            torch.save(model.state_dict(), open(args.model_path, "w"))
            # print input_unsorted.data.size()

            model.eval(input_unsorted, output_unsorted, args.train_output, args.training_set,
                       args.eval_output + ".train")
            model.eval(input_valid_unsorted, output_valid_unsorted, args.valid_output, args.valid_set,
                       args.eval_output + ".valid")
            model.eval(input_test_unsorted, output_test_unsorted, args.test_output, args.test_set,
                       args.eval_output + ".test")
        else:
            # print "neg_log_sum_loss for epoch " + str(epoch) + " " + str(query) + " " + str(k) + " " + str(
            #     neg_log_sum_loss.data[0])
            print "neg_log_sum_loss for epoch " + str(epoch) + " " + str(
                neg_log_sum_loss.data[0])


    else:
        print("Warning, loss goes up! new loss " + str(neg_log_sum_loss.data[0]) + " old " + str(prev_loss))

    # prev_loss[query] = neg_log_sum_loss.data[0]
    prev_loss = neg_log_sum_loss.data[0]
