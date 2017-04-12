import torch
import argparse
import torch.nn as nn
from torch.autograd import Variable
import utils
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
parser.add_argument('--lr', type=float, default=1 * 1e-3, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--load_model', type=bool, default=True, metavar='S',
                    help='are we loading pre-trained model?')

args = parser.parse_args()
torch.manual_seed(args.seed)

input_sorted, output_sorted, input_unsorted, output_unsorted, N, n, m = utils.load_data(
    args.training_set)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

input_test_sorted, output_test_sorted, input_test_unsorted, output_test_unsorted, N_test, n_test, m_test = utils.load_data(
    args.test_set)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

print "input sorted " + str(input_sorted)
# print input_sorted[0].data.size()
print "output sorted " + str(output_sorted)

print "input unsorted " + str(input_unsorted)
print "output unsorted " + str(output_unsorted)

print "N " + str(N)
print "n " + str(n)
print "m " + str(m)


class Net(nn.Module):
    def __init__(self, m):  # m is the feature dimension except x0 term
        super(Net, self).__init__()
        self.conv2 = nn.Conv2d(1, 1, kernel_size=(1, m), stride=(1, m),
                               bias=True)  # implicitly contains a learnable bias

        self.conv2.weight.data.zero_()
        self.conv2.bias.data.zero_()

        if args.load_model:
            self.load_state_dict(torch.load(args.model_path))
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
            score = scores[query].squeeze()  # e.g. score ordered by groundtruth label
            label = output[query].squeeze()  # e.g. label ordered by groundtruth, not existing doc as -1
            valid_doc_num = torch.sum((label > 0).data)
            exp_g = score.exp()
            # print "Exp_g " + str(exp_g)

            upper_limit_n = valid_doc_num
            # print  "upper limit n " + str(upper_limit_n)
            P_list = Variable(dtype(upper_limit_n))

            for i in range(upper_limit_n):
                P_list[i] = exp_g[i] / torch.sum(exp_g[i:upper_limit_n])
            # print P_list

            neg_log_sum -= sum(P_list.log())  # equation 9 in ListMLE paper
            # print sum(P_list.log())
            # print neg_log_sum
        return neg_log_sum

    def print_param(self):
        print [param.data for param in self.conv2.parameters()]

    def save_scores(self, scores_test, output_test, test_output_path):  # the feature list was unordered

        if os.path.exists(test_output_path):
            os.remove(test_output_path)

        for query in range(scores_test.size()[0]):
            label_test = output_test.data.squeeze()[query]
            valid_doc_num_test = sum(label_test > -1)
            scores_write = scores_test.data.squeeze()[query].numpy()[:valid_doc_num_test]
            np.savetxt(open(test_output_path, "a"), scores_write)


model = Net(m)

print args
prev_loss = float("inf")

original_lr = args.lr

optimizer = torch.optim.Adam(model.parameters(), lr=original_lr)

lr_scaled = False

ndcg = np.zeros(10)
precision = np.zeros(10)

for epoch in range(args.epochs):

    # Forward pass
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
            # print input_unsorted.data.size()
            scores_unsorted = model.forward(input_unsorted)
            # print scores_unsorted.data.size()
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
        # if lr_scaled:
        #     optimizer = torch.optim.Adam(model.parameters(), lr=original_lr*1.2)
        #     lr_scaled=False
        # model.conv2_prev_weight = model.conv2.weight.data.clone()
        # model.conv2_prev_bias = model.conv2.bias.data.clone()

    else:
        print("Warning, loss goes up! new loss " + str(neg_log_sum_loss.data[0]) + " old " + str(prev_loss))
        # optimizer = torch.optim.Adam(model.parameters(), lr=original_lr*0.1)
        # lr_scaled=True
        # original_lr=0.1*original_lr
        # model.conv2.weight.data = model.conv2_prev_weight
        # model.conv2.bias.data = model.conv2_prev_bias

    prev_loss = neg_log_sum_loss.data[0]
