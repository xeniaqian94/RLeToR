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

dtype = torch.FloatTensor

# pytorch example referece: https://github.com/jcjohnson/pytorch-examples/blob/master/autograd/two_layer_net_autograd.py
# regression/main.py, mnist/main.py

# nDCG calculation follows perl script Eval-Score-Score 3.0 from LeToR benchmark dataset, however, the formula was implemented
# slightly buggy from line 252 - 255. Position DCG was from index 1 instead of 0.

parser = argparse.ArgumentParser(description='PyTorch ListMLE Example')

# parser.add_argument('--training-set', type=str, default="../data/TD2003/Fold1/trainingset.txt",
#                     help='training set')
# parser.add_argument('--validation-set', type=str, default="../data/TD2003/Fold1/validationset.txt",
#                     help='validation set')
# parser.add_argument('--test-set', type=str, default="../data/TD2003/Fold1/testset.txt",
#                     help='test set')

parser.add_argument('--training_set', type=str, default="../data/toy/train.dat",
                    help='training set')
parser.add_argument('--validation_set', type=str, default="../data/toy/test.dat",
                    help='validation set')
parser.add_argument('--test_set', type=str, default="../data/toy/test.dat",
                    help='test set')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=172643969, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
torch.manual_seed(args.seed)

input, output, N, n, m = utils.load_data(
    args.training_set)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

input = Variable(torch.from_numpy(input).type(dtype), requires_grad=False)
output = Variable(torch.from_numpy(output).type(dtype), requires_grad=False)

print "input" + str(input)
print "output " + str(output)
print "N " + str(N)
print "n " + str(n)
print "m " + str(m)


class Net(nn.Module):
    def __init__(self, m):  # m is the feature dimension except x0 term
        super(Net, self).__init__()
        # self.filters = Variable(torch.randn(1, 1, 1, m + 1).type(dtype), requires_grad=True)
        # self.m_plus1 = m + 1
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, m + 1), stride=(1, m + 1))

    def forward(self, input):
        return self.conv2(input)

    def reset_grad(self):
        self.conv2.zero_grad()

    def seqMLELoss(self, scores, output):
        neg_log_sum = Variable(torch.zeros(1))
        for query in range(scores.size()[0]):
            print query
            score = scores[query].squeeze().data
            print score.size()  # e.g. score=[2.4500,1.412,0.929,-0.55]
            label = output[query].data  # e.g. label=[1,0,1,2]
            print label.size()

            order = torch.sort(label, 0, descending=True)[1]  # order=[3,0,2,1]
            order = torch.sort(order, 0)[
                1]  # order=[1,3,2,0] meaning score[0] -> position[1], score[1] -> position[3] ...

            ordered_score = dtype(score.size()[0])
            ordered_score.index_copy_(0, order, score)  # tensor copy based on the position order

            valid_doc_num = sum(label > -1)  # Documents that to be sorted in a query (not necessarily 100 per query)

            exp_g = ordered_score.exp()

            P_list = Variable(dtype(valid_doc_num))
            for i in range(valid_doc_num):
                P_list[i] = exp_g[i] / torch.sum(exp_g[i:valid_doc_num])

            neg_log_sum -= torch.prod(P_list, 0).log()  # equation 9 in ListMLE paper
        return neg_log_sum

    def print_param(self):
        print[param.data for param in self.conv2.parameters()]


model = Net(m)
model.reset_grad()

for epoch in range(args.epochs):
    # For batch training, always input and output

    # Forward pass
    print "Forward pass"
    model.print_param()
    scores = model.forward(input)  # N * n tensor
    print "score " + str(scores)
    print output

    # Reset gradients
    model.reset_grad()

    # Backward pass
    neg_log_sum_loss = model.seqMLELoss(scores, output)
    neg_log_sum_loss.backward()
    print neg_log_sum_loss

    print "Before gradient"
    print(model.conv1.parameters())
    # Apply gradients
    for param in model.conv1.parameters():
        print param.grad
        param.data.add_(-args.lr * param.grad.data)

    print "After gradient"
    print(model.conv1.parameters())

    # Stop criterion

    if neg_log_sum_loss < 1e-3:
        break


# print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
# print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
# print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
