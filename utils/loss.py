import torch
import torch.nn as nn


def triplet_loss(x, args):
    """

    :param x: 4*batch -> sk_p, sk_n, im_p, im_n
    :param args:
    :return:
    """
    triplet = nn.TripletMarginLoss(margin=1.0, p=2).cuda()
    sk_p = x[0:args.batch] #(b,768)
    im_p = x[2 * args.batch:3 * args.batch]
    im_n = x[3 * args.batch:]
    loss = triplet(sk_p, im_p, im_n)

    return loss


def rn_loss(predict, target):
    mse_loss = nn.MSELoss().cuda()
    loss = mse_loss(predict, target) #ipredict.shape=target.shape=(2b,1)

    return loss


def classify_loss(predict, target):
    class_loss = nn.CrossEntropyLoss().cuda()
    loss = class_loss(predict, target)

    return loss



