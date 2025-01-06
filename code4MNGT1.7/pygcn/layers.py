import math
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
# from models import resnet4GCN
from collections import OrderedDict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, head_num, out_features, image_size, patch_size, stride=1, padding=1, kernel_size=3, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = int(in_features/head_num)
        self.out_features = int(out_features/head_num)
        self.image_size = image_size
        self.patch_size = patch_size

        # 选择在图卷积中使用全连接层
        # (1)
        self.weight = Parameter(torch.FloatTensor(head_num, self.in_features, self.out_features).to(device))
        # randomatrix = torch.randn((head_num, self.in_features, self.out_features), requires_grad=True).to(device)
        # self.weight = torch.nn.Parameter(randomatrix)
        self.register_parameter('weight', self.weight)


        # (2)
        # self.head_num = head_num
        # Parameter = Parameter(torch.FloatTensor(int(in_features*head_num), int(out_features*head_num)))

        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_features).to(device))
            self.register_parameter('bias_gcn', self.bias)
        else:
            self.register_parameter('bias_gcn', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def norm(self, adj):
        head_num = adj.size(1)
        adj = rearrange(adj, 'b h n1 n2 -> (b h) n1 n2')
        adj = adj + torch.eye(adj.size(1), device=adj.device, dtype=adj.dtype)   # 为每个结点增加自环

        # degree = torch.pow(torch.einsum('ihjk->ihj', [adj]), -0.5)
        # degree[degree == float('inf')] = 0
        # degree_b = torch.eye(degree.size(2)).to(device)
        # degree_c = degree.unsqueeze(3).expand(*degree.size(), degree.size(2)).to(device)
        # degree_diag = degree_c * degree_b
        # norm_adj = degree_diag.matmul(adj).matmul(degree_diag).to(device)

        D = torch.diag_embed(torch.sum(adj, dim=-1) ** (-1 / 2))
        D[D == float('inf')] = 0
        norm_adj = torch.matmul(torch.matmul(D, adj), D)
        norm_adj = rearrange(norm_adj, '(b h) n1 n2 -> b h n1 n2', h=head_num)

        return norm_adj  # norm_adj

    def forward(self, input, adj):

        # 选择在图卷积中全连接层替换为卷积操作
        # (1)
        support = torch.matmul(input, self.weight)
        norm_adj = self.norm(adj)
        output = torch.matmul(norm_adj, support)  # spmm
        # if self.bias is not None:
        #     output = output + self.bias

        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
