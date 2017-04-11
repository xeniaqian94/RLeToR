import math
import torch
import argparse
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import utils
import re
import numpy as np
import os
import time

dtype = torch.FloatTensor

# pytorch example referece: https://github.com/jcjohnson/pytorch-examples/blob/master/autograd/two_layer_net_autograd.py
# regression/main.py, mnist/main.py

# nDCG calculation follows perl script Eval-Score-Score 3.0 from LeToR benchmark dataset, however, the formula was implemented
# slightly buggy from line 252 - 255. Position DCG was from index 1 instead of 0.

parser = argparse.ArgumentParser(description='PyTorch ListMLE Example')

# TD2003/Fold1
parser.add_argument('--training-set', type=str, default="../data/TD2003/Fold1/trainingset.txt",
                    help='training set')
# parser.add_argument('--validation-set', type=str, default="../data/TD2003/Fold1/validationset.txt",
#                     help='validation set')
parser.add_argument('--test-set', type=str, default="../data/TD2003/Fold1/testset.txt",
                    help='test set')
parser.add_argument('--test_output', type=str, default="../data/TD2003/Fold1/testoutput.txt",
                    help='test output')
parser.add_argument('--train_output', type=str, default="../data/TD2003/Fold1/trainoutput.txt",
                    help='train output')
parser.add_argument('--model_path', type=str, default="../data/TD2003/Fold1/model.txt",
                    help='model path')
parser.add_argument('--eval_output', type=str, default="../data/TD2003/Fold1/evaloutput.txt",
                    help='eval output path')
parser.add_argument('--list_cutoff', type=int, default=100, metavar='list_cutoff',
                    help='result list cutoff')

# toy example
# parser.add_argument('--training_set', type=str, default="../data/toy/train.dat",
#                     help='training set')
# # parser.add_argument('--validation_set', type=str, default="../data/toy/test.dat",
# #                     help='validation set')
# parser.add_argument('--test_set', type=str, default="../data/toy/test.dat",
#                     help='test set')
# parser.add_argument('--test_output', type=str, default="../data/toy/test_output.txt",
#                     help='test output')
# parser.add_argument('--train_output', type=str, default="../data/toy/train_output.txt",
#                     help='train output')
# parser.add_argument('--model_path', type=str, default="../data/toy/model.dat",
#                     help='model path')
# parser.add_argument('--eval_output', type=str, default="../data/toy/evaloutput.txt",
#                     help='eval output path')

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.1)')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
torch.manual_seed(args.seed)

input_sorted, output_sorted, input_unsorted, output_unsorted, N, n, m = utils.load_data(
    args.training_set)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

input_test_sorted, output_test_sorted, input_test_unsorted, output_test_unsorted, N_test, n_test, m_test = utils.load_data(
    args.test_set)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

print "input" + str(input_sorted)
print "output " + str(output_sorted)

print "input unsorted" + str(input_unsorted)
print "output unsorted" + str(output_unsorted)

print "N " + str(N)
print "n " + str(n)
print "m " + str(m)


class Net(nn.Module):
    def __init__(self, m):  # m is the feature dimension except x0 term
        super(Net, self).__init__()
        # self.filters = Variable(torch.randn(1, 1, 1, m + 1).type(dtype), requires_grad=True)
        # self.m_plus1 = m + 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, m), stride=(1, m),
                               bias=True)  # implicitly contains a learnable bias
        self.tanh = nn.Tanh()
        # self.conv2.weight.data.normal_(0, 0.1)

    def forward(self, input):

        return self.tanh(self.conv2(input))
        # return self.conv2(input)

    def reset_grad(self):
        self.conv2.zero_grad()

    def seqMLELoss(self, scores, output):
        neg_log_sum = Variable(torch.zeros(1))
        for query in range(scores.size()[0]):
            # for query in range(1):
            # print "query " + str(query)
            score = scores[query].squeeze()
            # print "score " + str(score)  # e.g. score=[2.4500,1.412,0.929,-0.55]
            label = output[query].squeeze()  # e.g. label=[1,0,1,2]
            # print "label " + str(label)
            # print score.size()
            # print label.size()

            # order = torch.sort(label, 0, descending=True)[1]  # order=[3,0,2,1]
            # order = torch.sort(order, 0)[
            #     1]  # order=[1,3,2,0] meaning score[0] -> position[1], score[1] -> position[3] ...
            #
            # ordered_score = dtype(score.size()[0])
            # ordered_score.index_copy_(0, order, score)  # tensor copy based on the position order



            # valid_doc_num = torch.sum((label > -1).data)  # Valid docs count (not necessarily 1000 per query)
            valid_doc_num = torch.sum((label > 0).data)*2
            # print valid_doc_num
            # exp_g = ordered_score.exp()
            # print torch.sort(score.data)
            exp_g = score.exp()
            # print exp_g
            # print torch.sort(exp_g.data)



            # upper_limit_n = min(valid_doc_num, args.list_cutoff)
            upper_limit_n = valid_doc_num
            # print  upper_limit_n
            P_list = Variable(dtype(upper_limit_n))

            for i in range(upper_limit_n):
                # if i % 50 == 0:
                #     print "query position" + str(query) + " " + str(i)
                # print "nominator " + str(exp_g[i])
                #
                # print "denominator " + str(torch.sum(exp_g[i:upper_limit_n]))
                # # print "denominator " + str(torch.sum(exp_g[i:10]))
                # print "torch.sum(exp_g) " + str(torch.sum(exp_g))
                # print str(exp_g[i] / torch.sum(exp_g[i:upper_limit_n]))
                P_list[i] = exp_g[i] / torch.sum(exp_g[i:upper_limit_n])

            # print torch.prod(P_list, 0).log()
            # print str(P_list)
            # print str(P_list.log())
            # print str(sum(P_list.log()))
            neg_log_sum -= sum(P_list.log())  # equation 9 in ListMLE paper
        return neg_log_sum

    def print_param(self):
        print[param.data for param in self.conv2.parameters()]

    def save_scores(self, scores_test, output_test, test_output_path):  # the feature list was unordered

        if os.path.exists(test_output_path):
            os.remove(test_output_path)

        for query in range(scores_test.size()[0]):
            label_test = output_test.data.squeeze()[query]
            valid_doc_num_test = sum(label_test > -1)
            # print valid_doc_num_test
            scores_write = scores_test.data.squeeze()[query].numpy()[:valid_doc_num_test]
            # print scores_valid.shape
            np.savetxt(open(test_output_path, "a"), scores_write)
        print "save finished"


model = Net(m)

# model.reset_grad()

print args
prev_loss = float("inf")
# print (input[0][0][376])
# print (input[0][0][0])

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

ndcg = np.zeros(10)
precision = np.zeros(10)

for epoch in range(args.epochs):
    # For batch training, always input and output

    # Forward pass
    # print "Forward pass"
    # model.print_param()
    scores = model.forward(input_sorted)  # N * n tensor
    # print "score " + str(scores)
    # print output

    # Reset gradients
    # model.reset_grad()
    optimizer.zero_grad()

    # Backward pass
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
        # model.print_param()
        # Save the model see discussion: https: // discuss.pytorch.org / t / saving - torch - models / 838 / 4

        if abs(neg_log_sum_loss.data[0] - prev_loss) < 1e-5:
            torch.save(model.state_dict(), open(args.model_path, "w"))
            # scores_test = model.forward(input_test)
            # model.save_scores(scores_test, output_test, args.test_output)

            break
        if (epoch % 20 == 0):
            torch.save(model.state_dict(), open(args.model_path, "w"))
            print input_unsorted.data.size()
            scores_unsorted = model.forward(input_unsorted)
            print scores_unsorted.data.size()
            model.save_scores(scores_unsorted, output_unsorted, args.train_output)
            os.system(
                "perl Eval-Score-3.0.pl " + args.training_set + " " + args.train_output + " " + args.eval_output + " 0")

            with open(args.eval_output, "r") as eval_f:
                for line in eval_f.readlines():
                    if ("precision:	") in line:
                        precision = [float(value) for value in line.split()[1:11]]
                    elif ("MAP:	") in line:
                        map = float(line.split()[1])
                    elif ("NDCG:	") in line:
                        ndcg = [float(value) for value in line.split()[1:11]]

            print "PLOT Epoch " + str(epoch) + " " + str(neg_log_sum_loss.data[0]) + " ndcg " + str(
                ndcg) + " map " + str(map) + " precision " + str(precision)

        else:
            print "neg_log_sum_loss for epoch " + str(epoch) + " " + str(neg_log_sum_loss.data[0])



    else:
        print("Warning, loss goes up! new loss " + str(neg_log_sum_loss.data[0]) + " old " + str(prev_loss))

    prev_loss = neg_log_sum_loss.data[0]
