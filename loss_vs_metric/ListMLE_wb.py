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

parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=172643969, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
torch.manual_seed(args.seed)

input, output, N, n, m = utils.load_data_ListMLE(
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
        print("Feature dimension except x0 :" + str(m))
        self.W_target = torch.randn(m + 1)
        self.W_target = Variable(self.W_target.type(dtype), requires_grad=True)

    def forward(self, x):
        # v_view = v.unsqueeze(0).expand(-1, len(v)).unsqueeze(2)  # batch x dim2 x 1
        return x * self.W_target

    def reset_grad(self):
        print self.W_target.grad



model = Net(m)

for epoch in range(args.epochs):
    # For batch training, always input and output
    # Reset gradients
    print("initial weight")
    print(model.W_target)
    scores = model.forward(input)
    print scores
    model.reset_grad()



    # model.forward(x)

    # Forward pass
#     output = F.smooth_l1_loss(fc(batch_x), batch_y)
#     loss = output.data[0]
#
#     # Backward pass
#     output.backward()
#
#     # Apply gradients
#     for param in fc.parameters():
#         param.data.add_(-0.1 * param.grad.data)
#
#     # Stop criterion
#     if loss < 1e-3:
#         break
#
# print('Loss: {:.6f} after {} batches'.format(loss, batch_idx))
# print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
# print('==> Actual function:\t' + poly_desc(W_target.view(-1), b_target))
