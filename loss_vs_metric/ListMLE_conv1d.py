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
parser.add_argument('--model_path', type=str, default="../data/TD2003/Fold1/model.txt",
                    help='model path')


# toy example
# parser.add_argument('--training_set', type=str, default="../data/toy/train.dat",
#                     help='training set')
# parser.add_argument('--validation_set', type=str, default="../data/toy/test.dat",
#                     help='validation set')
# parser.add_argument('--test_set', type=str, default="../data/toy/test.dat",
#                     help='test set')
# parser.add_argument('--test_output', type=str, default="../data/toy/test_output.txt",
#                     help='test output')
# parser.add_argument('--model_path', type=str, default="../data/toy/model.dat",
#                     help='model path')

parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1, metavar='LR',
                    help='learning rate (default: 0.1)')

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

input_test, output_test, N_test, n_test, m_test = utils.load_data(
    args.test_set)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

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
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, m), stride=(1, m),
                               bias=True)  # implicitly contains a learnable bias

    def forward(self, input):
        return self.conv2(input)

    def reset_grad(self):
        self.conv2.zero_grad()

    def seqMLELoss(self, scores, output):
        neg_log_sum = Variable(torch.zeros(1))
        for query in range(scores.size()[0]):
            # print "query " + str(query)
            score = scores[query].squeeze()
            # print "score " + str(score)  # e.g. score=[2.4500,1.412,0.929,-0.55]
            label = output[query]  # e.g. label=[1,0,1,2]
            # print "label " + str(label)

            # order = torch.sort(label, 0, descending=True)[1]  # order=[3,0,2,1]
            # order = torch.sort(order, 0)[
            #     1]  # order=[1,3,2,0] meaning score[0] -> position[1], score[1] -> position[3] ...
            #
            # ordered_score = dtype(score.size()[0])
            # ordered_score.index_copy_(0, order, score)  # tensor copy based on the position order

            valid_doc_num = sum(label > -1)  # Documents that to be sorted in a query (not necessarily 100 per query)
            # print valid_doc_num
            # exp_g = ordered_score.exp()
            exp_g = score.exp()
            # print exp_g

            P_list = Variable(dtype(valid_doc_num.data[0]))

            for i in range(valid_doc_num.data[0]):
                P_list[i] = exp_g[i] / torch.sum(exp_g[i:valid_doc_num.data[0]])
            # print P_list
            neg_log_sum -= torch.prod(P_list, 0).log()  # equation 9 in ListMLE paper
        return neg_log_sum

    def print_param(self):
        print[param.data for param in self.conv2.parameters()]

    def save_scores(self, scores_test, output_test, test_output_path):
        if os.path.exists(test_output_path):
            os.remove(test_output_path)

        for query in range(scores_test.size()[0]):
            label = output_test[query]
            valid_doc_num = sum(label > -1)
            scores_valid = scores_test[query].data.squeeze().numpy()[:valid_doc_num.data[0]]
            # print scores_valid.shape
            np.savetxt(open(test_output_path, "a"), scores_valid)


model = Net(m)
model.reset_grad()
print args
prev_loss = float("inf")

for epoch in range(args.epochs):
    # For batch training, always input and output

    # Forward pass
    # print "Forward pass"
    # model.print_param()
    scores = model.forward(input)  # N * n tensor
    # print "score " + str(scores)
    # print output

    # Reset gradients
    model.reset_grad()

    # Backward pass
    neg_log_sum_loss = model.seqMLELoss(scores, output)
    neg_log_sum_loss.backward()
    print "neg_log_sum_loss for epoch " + str(epoch) + " " + str(neg_log_sum_loss.data[0])

    # print "Before gradient"
    # Apply gradients
    for param in model.conv2.parameters():
        param.data.add_(-args.lr * param.grad.data)

    # print "After gradient"
    # model.print_param()

    # Stop criterion


    if neg_log_sum_loss.data[0] < prev_loss:
        # model.print_param()
        # Save the model see discussion: https: // discuss.pytorch.org / t / saving - torch - models / 838 / 4


        if abs(neg_log_sum_loss.data[0] - prev_loss) < 1e-3:
            torch.save(model.state_dict(), open(args.model_path, "w"))
            scores_test = model.forward(input_test)
            model.save_scores(scores_test, output_test, args.test_output)
            break
        if (epoch % 10 == 0):
            torch.save(model.state_dict(), open(args.model_path, "w"))
    else:
        print("Warning, loss goes up!")

    prev_loss = neg_log_sum_loss.data[0]


# print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
# print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
# print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
