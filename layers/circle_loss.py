# encoding: utf-8
"""
@author:  suqinghang
@contact: qinghang_su@163.com
"""
import torch
from torch import nn, Tensor

class CircleLoss(nn.Module):
    '''Circle loss.

    Reference:
    Sun et al. Circle Loss: A Unified Perspective of Pair Similarity Optimization. CVPR 2020

    Modified from TinyZeaMays's CircleLoss(https://github.com/TinyZeaMays/CircleLoss)

    '''
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, x, labels, normalization=False):
        '''

        :param x: feature matrix with shape(batch_size, feat_dim).
        :param labels:  ground truth labels with shape (num_classes).
        :return: circle loss
        '''
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        if normalization:
            x = nn.functional.normalize(x)
        similarity_matrix = x@x.transpose(1,0)
        labels_matrix = labels.unsqueeze(1) == labels.unsqueeze(0)

        positive_matrix = labels_matrix.triu(diagonal=1)
        negative_matrix = labels_matrix.logical_not().triu(diagonal=1)
        similarity_matrix = similarity_matrix.view(-1)
        positive_matrix = positive_matrix.view(-1)
        negative_matrix = negative_matrix.view(-1)

        sp = similarity_matrix[positive_matrix]
        sn = similarity_matrix[negative_matrix]

        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0)
        an = torch.clamp_min(sn.detach() + self.m, min=0)

        delta_p = 1-self.m
        delta_n = self.m

        logit_p = - ap + (sp - delta_p) * self.gamma
        logit_n = an * (sn -  delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
        return loss
