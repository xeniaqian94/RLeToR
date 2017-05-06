'''
    File name: reinforce
    Author: xin
    Date created: 5/1/17 1:10 PM
'''

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

dtype = torch.FloatTensor

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
# parser.add_argument('--random_level', type=int, default=1,
#                     help="random level 1 = randomly select one ground-truth throughout training, random level 2 = dynamically & randomly select ground-truth every epoch ")
# parser.add_argument('--iter', type=int, default=1,
#                     help="iteration")
parser.add_argument('--model_path', type=str, default="../data/TD2003/Fold1/model.txt",
                    help='model path')
parser.add_argument('--load_model_path', type=str, default="../data/TD2003/Fold1/mode",
                    help='model path')
parser.add_argument('--eval_output', type=str, default="../data/TD2003/Fold1/evaloutput.txt",
                    help='eval output path')
# parser.add_argument('--list_cutoff', type=int, default=100, metavar='list_cutoff',
#                     help='result list cutoff')


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
# parser.add_argument('--eval_output', type=str, default="../data/toy/evaloutput.txt.reinforce",
#                     help='eval output path')

parser.add_argument('--reward_metric', type=str, default="NDCG",
                    help='which metric to user as immediate reward')

parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.1)')
parser.add_argument('--discount', type=float, default=1.0, metavar='discount',
                    help='discount rate')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--load_model', type=bool, default=True, metavar='S',
                    help='whether to load pre-trained model?')
parser.add_argument('--query_dimension_normalization', type=bool, default=True, metavar='S',
                    help='whether to normalize by query-dimension?')

args = parser.parse_args()

torch.manual_seed(args.seed)

input_sorted, output_sorted, input_unsorted, output_unsorted, N, n, m = utils.load_data_ListMLE(
    args.training_set,
    args.query_dimension_normalization)  # N # of queries, n # document per query, m feature dimension (except the x_0 term)

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
        self.softmax = nn.Softmax()

        if args.load_model and os.path.exists(args.load_model_path):
            self.load_state_dict(torch.load(args.load_model_path))
            print "Pre-trained model loaded"

    def forward(self, input):  # online ranking
        return self.conv2(input)

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

    def eval(self, input_this, output_this, outputpath_this, trainpath_this, evalpath_this,neg_log_sum_loss):
        scores_this = model.forward(input_this)

        model.save_scores(scores_this, output_this, outputpath_this)  # scores this first assign highest score

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

                for i in range(upper_limit_n):
                    P_list[i] = exp_g[i] / torch.sum(exp_g[i:valid_doc_num])  # one-hot groundtruth

                neg_log_sum -= sum(P_list.log())  # equation 9 in ListMLE paper

        return neg_log_sum


def eval_reward(pred_scores, groundtruth_scores, predpath_this, trainpath_this, evalpath_this):
    reward = torch.zeros(groundtruth_scores.data.size())

    model.save_scores(pred_scores, groundtruth_scores, predpath_this)  # scores this first assign highest score

    os.system(
        "perl Eval-Score-3.0.pl " + trainpath_this + " " + predpath_this + " " + evalpath_this + " 1")

    with open(evalpath_this, "r") as eval_f:
        for line in eval_f.readlines():

            if (args.reward_metric + " of ") in line:
                query_ind = int(re.search(r"of query([0-9]+):", line).group(1))
                metrics_level=min(len(line.split(":")[1].split()), reward.size()[1])
                reward[query_ind][:metrics_level] = torch.from_numpy(np.asarray(
                    [float(value) for value in
                     line.split(":")[1].split()[:metrics_level]]))
                # print torch.from_numpy(np.asarray(
                #     [float(value) for value in
                #      line.split(":")[1].split()[:metrics_level]]))

    return reward


def sample_episode(model, input_unsorted, output_unsorted):  # batch update or SGD?
    scores = model.forward(input_unsorted)
    order_as_score = Variable(torch.zeros(scores.squeeze().data.size()))

    log_pi = Variable(torch.zeros(scores.squeeze().data.size()))

    # masks=Variable(torch.zeros(log_pi.size()[0],log_pi.size()[1],log_pi.size()[1]))
    selected_doc=Variable(torch.zeros(log_pi.size()[0],log_pi.size()[1]))

    for query in range(scores.size()[0]):
        torch.manual_seed(int(time.time()))
        M = torch.sum((output_unsorted[query] > -1).data)
        score_exp = scores[query].squeeze().exp()  # [torch.FloatTensor of size nx1]


        masks=Variable(torch.cat((torch.ones(M), torch.zeros(n - M))).unsqueeze(0).repeat(M,1),requires_grad=False)
        # masks[query,:M,:n] = Variable(torch.cat((torch.ones(M), torch.zeros(n - M))).expand(M, n),requires_grad=False)
        # masks_clone=Variable(torch.Tensor(M,n))
        selected=[]
        print query,masks

        for t in range(M):
            # print scores[:20]
            # print query, t, torch.sum(masks[t]),torch.sum(score_exp * masks[t])<1e-45,torch.sum(score_exp * masks[t])==0

            # selected_doc_id = torch.multinomial(score_exp * masks[query,t,:score_exp.size()[0]]).data.numpy()[0]

            selected_doc_id = torch.multinomial(score_exp * masks[t]).data.numpy()[0]

            selected+=[selected_doc_id]    # ranking docid list
            # selected_doc[query,t]=selected_doc_id

            order_as_score[query, selected_doc_id] = M - t

            if t < M - 1:
                masks[t + 1]=Variable(torch.from_numpy(masks[t].data.clone().numpy()))
                masks[t+1,selected_doc_id]=0
                print query, t,masks
            # if t < M - 1:
            #     masks[query, t+1, :score_exp.size()[0]]=masks[query,t,:score_exp.size()[0]].clone()
            #     masks[query, t+1, selected_doc_id]=0

        masks_clone=masks.clone()
        print query, masks
        # print "masks_clone 0 "+str(masks_clone[0])
        for t in range(M):
            log_pi[query, t] = torch.log(score_exp[selected[t]] / torch.sum(score_exp * masks_clone[t]))
            # print query,t, score_exp[selected[t]].data.numpy(), torch.sum(score_exp * masks_clone[t]).data.numpy(),score_exp[selected[t]] / torch.sum(score_exp * masks_clone[t])

    # masks_clone=masks.clone()
    reward = eval_reward(order_as_score, output_unsorted, args.training_set + ".episode.pred", args.training_set,
                         args.training_set + ".episode.eval")

    print str(log_pi)
    return reward, log_pi
    # return reward,masks_clone,selected_doc


epoch = 0


def MDP_for_gradient(model, input_unsorted, output_unsorted):  # equation 4
    reward, log_pi = sample_episode(model, input_unsorted, output_unsorted)
    # reward,masks,selected_doc=sample_episode(model, input_unsorted, output_unsorted)
    sum=Variable(torch.Tensor(1))
    model.zero_grad()

    for query in range(output_unsorted.size()[0]):
        M = torch.sum((output_unsorted[query] > -1).data)  # valid docs

        for t in range(M):
            Gt = reward[query, M - 1] - reward[query, t]  # equation 4 when discount = 1.0


            # print query, t, str(Gt),reward[query, M - 1],reward[query, t]
            # model.zero_grad()
            # score_exp=model(input_unsorted[query].unsqueeze(0)).squeeze().exp()
            # to_gradient=Gt*torch.log(score_exp[selected_doc[query,t].data.numpy()[0]]/torch.sum(score_exp*masks[query,t]))
            # to_gradient.backward()
            # for param in model.parameters():
            #     param.data += args.lr * param.grad.data
            # epoch+=1
            #
            # if epoch%2==0:
            #     scores = model.forward(input_sorted)  # N * n tensor
            #
            #     neg_log_sum_loss = model.seqMLELoss(scores, output_sorted)
            #
            #     if (epoch % 2 == 0):
            #         torch.save(model.state_dict(), open(args.model_path, "w"))
            #         model.eval(input_unsorted, output_unsorted, args.train_output, args.training_set,
            #                    args.eval_output + ".train",neg_log_sum_loss)
            #         model.eval(input_valid_unsorted, output_valid_unsorted, args.valid_output, args.valid_set,
            #                    args.eval_output + ".valid",neg_log_sum_loss)
            #         model.eval(input_test_unsorted, output_test_unsorted, args.test_output, args.test_set,
            #                    args.eval_output + ".test",neg_log_sum_loss)

            sum += Gt * log_pi[query, t]  #one of the variables needed for gradient computation has been modified by an inplace operation

    sum.backward()

    for param in model.parameters():
        param.data += args.lr * param.grad.data




model = Net(m)

print args

# model.eval(input_unsorted, output_unsorted, args.train_output, args.training_set, args.eval_output + ".train")
# model.eval(input_valid_unsorted, output_valid_unsorted, args.valid_output, args.valid_set,
#            args.eval_output + ".valid")
# model.eval(input_test_unsorted, output_test_unsorted, args.test_output, args.test_set,
#            args.eval_output + ".test")

fold_num = int(re.search(r"Fold([0-9]+)/", args.model_path).group(1))
prev_loss=float("inf")
while True:
    MDP_for_gradient(model, input_unsorted, output_unsorted)
    epoch+=1

    if epoch % 2 == 0:
        scores = model.forward(input_sorted)  # N * n tensor

        neg_log_sum_loss = model.seqMLELoss(scores, output_sorted)

        if (epoch % 2 == 0):
            torch.save(model.state_dict(), open(args.model_path, "w"))
            model.eval(input_unsorted, output_unsorted, args.train_output, args.training_set,
                       args.eval_output + ".train", neg_log_sum_loss)
            model.eval(input_valid_unsorted, output_valid_unsorted, args.valid_output, args.valid_set,
                       args.eval_output + ".valid", neg_log_sum_loss)
            model.eval(input_test_unsorted, output_test_unsorted, args.test_output, args.test_set,
                       args.eval_output + ".test", neg_log_sum_loss)





