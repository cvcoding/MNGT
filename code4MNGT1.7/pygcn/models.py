import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc2 = GraphConvolution(nfeat, head_num, nhid, image_size, patch_size, stride, padding, kernel_size)
        # self.gc3 = GraphConvolution(nfeat, nhid, image_size, patch_size, stride, padding, kernel_size)
        self.dropout = dropout
        # self.linear = nn.Linear(int(nfeat/head_num), int(nhid/head_num))

    def forward(self, x, adj):
        x = F.gelu(self.gc1(x, adj))  # Having GELU
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.gelu(self.gc2(x, adj))

        # x = self.linear(x)
        return x
