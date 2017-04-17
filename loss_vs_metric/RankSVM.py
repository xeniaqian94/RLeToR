import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import utils
import numpy as np
import os
import sys

dtype = torch.FloatTensor

parser = argparse.ArgumentParser(description='PyTorch RankSVM Example')

# TD2003/Fold1
# parser.add_argument('--training_set', type=str, default="../data/TD2003/Fold1/trainingset.txt",
#                     help='training set')
# parser.add_argument('--valid_set', type=str, default="../data/TD2003/Fold1/validationset.txt",
#                     help='validation set')
# parser.add_argument('--test_set', type=str, default="../data/TD2003/Fold1/testset.txt",
#                     help='test set')
# parser.add_argument('--test_output', type=str, default="../data/TD2003/Fold1/testoutput.txt",
#                     help='test output')
# parser.add_argument('--train_output', type=str, default="../data/TD2003/Fold1/trainoutput.txt",
#                     help='train output')
# parser.add_argument('--valid_output', type=str, default="../data/TD2003/Fold1/validoutput.txt",
#                     help='valid output')
#
# parser.add_argument('--model_path', type=str, default="../data/TD2003/Fold1/model.txt",
#                     help='model path')
# parser.add_argument('--eval_output', type=str, default="../data/TD2003/Fold1/evaloutput.txt",
#                     help='eval output path')
# parser.add_argument('--list_cutoff', type=int, default=100, metavar='list_cutoff',
#                     help='result list cutoff')

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
parser.add_argument('--training_set', type=str, default="../data/toy/train.dat",
                    help='training set')
parser.add_argument('--valid_set', type=str, default="../data/toy/test.dat",
                    help='validation set')
parser.add_argument('--test_set', type=str, default="../data/toy/test.dat",
                    help='test set')
parser.add_argument('--test_output', type=str, default="../data/toy/test_output.txt",
                    help='test output')
parser.add_argument('--train_output', type=str, default="../data/toy/train_output.txt",
                    help='train output')
parser.add_argument('--valid_output', type=str, default="../data/toy/test_output.txt",
                    help='valid output')
parser.add_argument('--model_path', type=str, default="../data/toy/model.dat",
                    help='model path')
parser.add_argument('--eval_output', type=str, default="../data/toy/evaloutput.txt",
                    help='eval output path')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--c', type=float, default=0.1, metavar='c',
                    help='trading-off margin size against training error')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load_model', type=bool, default=False, metavar='S',
                    help='whether to load pre-trained model?')
parser.add_argument('--query_dimension_normalization', type=bool, default=True, metavar='S',
                    help='whether to normalize by query-dimension?')

args = parser.parse_args()
torch.manual_seed(args.seed)

input, output, input_pos, input_neg, output_pos, output_neg, N, n1, n2, m = utils.load_data_RankSVM(
    args.training_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

input_test, output_test, input_test_pos, input_test_neg, output_test_pos, output_test_neg, N_test, n1_test, n2_test, m_test = utils.load_data_RankSVM(
    args.test_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

input_valid, output_valid, input_valid_pos, input_valid_neg, output_valid_pos, output_valid_neg, N_valid, n1_valid, n2_valid, m_valid = utils.load_data_RankSVM(
    args.valid_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

print "input positive" + str(input_pos)
print "input negative " + str(input_neg)

print "output positive" + str(output_pos)
print "output negative " + str(output_neg)

print "N " + str(N)
print "n1 " + str(n1)
print "n2 " + str(n2)
print "m " + str(m)
print input.data.size()
print output.data.size()


class Net(nn.Module):
    def __init__(self, m):  # m is the feature dimension except x0 term
        super(Net, self).__init__()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, m), stride=(1, m),
                               bias=False)  # implicitly contains a learnable bias

        # self.conv2.weight.data.zero_()

        if args.load_model and os.path.exists(args.model_path):
            self.load_state_dict(torch.load(args.model_path))
            print "Pre-trained model loaded"

    def forward(self, input):
        # return self.tanh(self.conv2(input))
        return self.conv2(input)

    def pairwiseHinge(self, scores_pos, scores_neg, output_pos, output_neg):
        slack = Variable(torch.zeros(1))

        # print scores_pos
        # print scores_neg

        for query in range(scores_pos.size()[0]):
            pos_score = scores_pos[query].squeeze()
            pos_label = output_pos[query].squeeze()
            neg_score = scores_neg[query].squeeze()
            neg_label = output_neg[query].squeeze()
            for i in range(torch.sum((pos_label > -1).data)):
                for j in range(torch.sum((neg_label > -1).data)):
                    slack += torch.max(1 - (pos_score[i] - neg_score[j]), Variable(torch.Tensor([0])))
                    # print slack

        return 0.5 * torch.sum(torch.pow(self.conv2.weight, 2)) + args.c * slack

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

    def eval(self, input_this, output_this, outputpath_this, trainpath_this, evalpath_this):
        scores_this = model.forward(input_this)  # Both positive and negative examples
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

        print "PLOT Epoch " + str(epoch) + " " + trainpath_this + " " + str(pairwiseHingeLoss.data[0]) + " ndcg " + str(
            ndcg) + " map " + str(map) + " precision " + str(precision)


model = Net(m)

print args
# prev_loss = [float('inf') for i in range(input_sorted.data.size()[0])]
prev_loss = float('inf')
original_lr = args.lr

optimizer = torch.optim.Adam(model.parameters(), lr=original_lr)

ndcg = np.zeros(10)
precision = np.zeros(10)

epoch = -1

pairwiseHingeLoss = Variable(torch.zeros(1))

model.eval(input, output, args.train_output, args.training_set, args.eval_output + ".train")
model.eval(input_valid, output_valid, args.valid_output, args.valid_set,
           args.eval_output + ".valid")
model.eval(input_test, output_test, args.test_output, args.test_set,
           args.eval_output + ".test")

for epoch in range(args.epochs):

    scores_pos = model.forward(input_pos)  # N * n tensor
    scores_neg = model.forward(input_neg)

    optimizer.zero_grad()

    model.print_param()

    # print scores_pos.data.size(), scores_neg.data.size()
    pairwiseHingeLoss = model.pairwiseHinge(scores_pos, scores_neg, output_pos, output_neg)

    pairwiseHingeLoss.backward()

    optimizer.step()

    if pairwiseHingeLoss.data[0] < prev_loss:
        # if neg_log_sum_loss.data[0] < prev_loss[query]:
        model.print_param()
        # Save the model see discussion: https: // discuss.pytorch.org / t / saving - torch - models / 838 / 4

        if abs(pairwiseHingeLoss.data[0] - prev_loss) < 1e-5:
            # if abs(neg_log_sum_loss.data[0] - prev_loss[query]) < 1e-5:
            torch.save(model.state_dict(), open(args.model_path, "w"))
            break
        if (epoch % 2 == 0):
            # print model.print_param()
            # print "input unsorted "+str(input_unsorted)
            torch.save(model.state_dict(), open(args.model_path, "w"))
            # print input_unsorted.data.size()
            model.eval(input, output, args.train_output, args.training_set, args.eval_output + ".train")
            model.eval(input_valid, output_valid, args.valid_output, args.valid_set,
                       args.eval_output + ".valid")
            model.eval(input_test, output_test, args.test_output, args.test_set,
                       args.eval_output + ".test")

        else:
            # print "neg_log_sum_loss for epoch " + str(epoch) + " " + str(query) + " " + str(k) + " " + str(
            #     neg_log_sum_loss.data[0])
            print "pairwiseHingeLoss for epoch " + str(epoch) + " " + str(
                pairwiseHingeLoss.data[0])

    else:
        print("Warning, loss goes up! new loss " + str(pairwiseHingeLoss.data[0]) + " old " + str(prev_loss))
    sys.stdout.flush()

    # prev_loss[query] = neg_log_sum_loss.data[0]
    prev_loss = pairwiseHingeLoss.data[0]
