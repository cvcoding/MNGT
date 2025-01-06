# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from functools import partial
from itertools import repeat
from collections import OrderedDict
from einops import rearrange
from einops.layers.torch import Rearrange
import math
from pygcn.models import GCN
from pygcn.models_gru import GCN_gru
import numpy as np
# import cupy as cp
# from models import mobilenet

from models import *
from torch.autograd import Variable
import torchvision.transforms as transforms

dtype = torch.cuda.FloatTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# from torch._six import container_abcs

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        # if isinstance(x, container_abcs.Iterable):
        #     return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)

MIN_NUM_PATCHES = 16


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        att, adj = self.fn(x, *args, **kwargs)
        return att + x, adj


class Residual_out(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, x_resize, *args, **kwargs):
        out, out_resize, rep_adj, rep_adj_resize = self.fn(x, x_resize, *args, **kwargs)

        return out + x, out_resize + x_resize, rep_adj, rep_adj_resize


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        temp = self.norm(x)
        return self.fn(temp, *args, **kwargs)


class PreNorm_out(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, x_resize, *args, **kwargs):
        temp = self.norm(x)
        temp1 = self.norm(x_resize)
        return self.fn(temp, temp1, *args, **kwargs)


# class PreForward(nn.Module):
#     def __init__(self, dim, hidden_dim, kernel_size, num_channels, dropout=0.):
#         super().__init__()
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         #     nn.Linear(hidden_dim, dim),
#         #     nn.Dropout(dropout)
#         # )
#         self.tcn = TemporalConvNet(dim, num_channels, hidden_dim, kernel_size, dropout)
#         # self.net = nn.Sequential(
#         #     nn.Linear(dim, hidden_dim),
#         #     nn.GELU(),
#         #     nn.Dropout(dropout),
#         # )
#
#     def forward(self, x):
#         r = self.tcn(x.permute(0, 2, 1)).permute(0, 2, 1)
#         # r = self.net(r)
#         return r


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, image_size, patch_size, dropout=0.):
        super().__init__()
        # self.net = nn.Sequential(
        #     nn.Linear(dim, hidden_dim),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     # nn.Linear(hidden_dim, dim),
        #     # nn.Dropout(dropout)
        # )
        self.net = nn.Identity()

    def forward(self, x):
        return self.net(x)


# def inverse_gumbel_cdf(y, mu, beta):
#     return mu - beta * torch.log(-torch.log(y))


class Attention(nn.Module):
    def __init__(self,
                 depth,
                 i,
                 dim,
                 image_size,
                 patch_size,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.1,
                 proj_drop=0.,
                 downsample=0.,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False
                 ):
        super().__init__()
        self.scale = dim ** -0.5
        # self.drop_ratio = 0.1

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.with_cls_token = with_cls_token

        self.length = int(image_size / patch_size) ** 2
        self.length2 = int(image_size / patch_size * downsample) ** 2

        self.layer_index = i
        self.depth = depth

        dim_in = dim
        dim_out = dim

        self.conv_proj_q = GCN(nfeat=dim_in,
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)
        self.conv_proj_k = GCN(nfeat=dim_in,
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)
        self.conv_proj_v = GCN(nfeat=dim_in,  # GCN_gru
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        # self.proj_q.weight.requires_grad = False
        # self.proj_k.weight.requires_grad = False
        # self.proj_v.weight.requires_grad = False

        # self.attn_drop = nn.Dropout(attn_drop)  #
        # self.proj_drop = nn.Dropout(proj_drop)  #
        self.leakyrelu = nn.LeakyReLU()

        # sparse_D = torch.ones((self.num_heads, self.length), requires_grad=True).to(device)
        # self.sparse_D = torch.nn.Parameter(sparse_D)
        # self.register_parameter("sparse_D", self.sparse_D)
        #
        # sparse_D2 = torch.ones((self.num_heads, self.length2), requires_grad=True).to(device)
        # self.sparse_D2 = torch.nn.Parameter(sparse_D2)
        # self.register_parameter("sparse_D2", self.sparse_D2)

        randomatrix = torch.randn((int(self.num_heads),
                                   int(self.num_heads)), requires_grad=True).to(device)
        self.randomatrix = torch.nn.Parameter(randomatrix)
        self.register_parameter("Ablah", self.randomatrix)

        # from torch.nn.parameter import Parameter
        # self.randomatrix = Parameter(torch.FloatTensor(int(self.num_heads), int(self.num_heads))).to(device)

        # self.proj_k_f = nn.Linear(2 * dim_in, self.num_heads, bias=qkv_bias)
        # self.proj_k_f2 = nn.Linear(self.num_heads, self.length, bias=qkv_bias)
        # self.proj_k_f3 = nn.Linear(self.num_heads, self.length2, bias=qkv_bias)
        # self.proj_k_f.weight.requires_grad = False
        # self.proj_k_f2.weight.requires_grad = False
        # self.proj_k_f3.weight.requires_grad = False

    def forward_conv_qk(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x, rep_adj)
            # q = F.dropout(F.relu(q), self.drop_ratio, training=self.training)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x, rep_adj)
            # k = F.dropout(F.relu(k), self.drop_ratio, training=self.training)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        return q, k

    def forward_conv_v(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x, rep_adj)
            # v = F.dropout(v, self.drop_ratio, training=self.training)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return v

    def forward(self, x, adj, label):
        x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
        current_length = adj.size(-1)
        b, head, num_rows, embedding_dim = x.size()

        # repeat_x = x.unsqueeze(3).cpu()
        # repeat_x_T = x.unsqueeze(2).cpu()
        # similarity = 1 - torch.cosine_similarity(repeat_x, repeat_x_T, dim=-1)
        # similarity = torch.softmax(similarity.to(device), dim=-1)
        # adj4qk = similarity.matmul(adj).matmul(similarity)

        # new_b = int(b*head)
        # matrices = rearrange(x, 'b h t d -> (b h) t d')
        # expanded_matrices = matrices.unsqueeze(1)  # Shape: (batch_size, 1, num_rows, embedding_dim)
        # replicated_matrices = expanded_matrices.repeat(1, num_rows, 1,1)  # Shape: (batch_size, num_rows, num_rows, embedding_dim)
        # flattened_replicated_matrices = replicated_matrices.view(-1, num_rows, embedding_dim)  # Shape: (batch_size * num_rows, num_rows, embedding_dim)
        # cosine_similarities = F.cosine_similarity(flattened_replicated_matrices, flattened_replicated_matrices, dim=2)
        # cosine_similarities = cosine_similarities.view(new_b, num_rows, num_rows)
        # cos_sim_matrix = rearrange(cosine_similarities, '(b h) t d->b h t d', h=self.num_heads)
        # adj4qk = cos_sim_matrix.matmul(adj).matmul(cos_sim_matrix)

        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k = self.forward_conv_qk(x, adj)

        # q = ((rearrange(q, 'b h t d -> b t (h d)', h=head)))
        # k = ((rearrange(k, 'b h t d -> b t (h d)', h=head)))

        # qk = torch.concat((q, k), dim=-1)
        # Random_RM0 = F.gelu(self.proj_k_f(qk))
        # Random_RM = torch.matmul(Random_RM0.permute(0, 2, 1), Random_RM0)
        #
        # if label == 0:
        #     Random_RM = torch.sigmoid(self.proj_k_f2(Random_RM) / self.sparse_D)
        # else:
        #     Random_RM = torch.sigmoid(self.proj_k_f3(Random_RM) / self.sparse_D2)

        # Random_RM = torch.diag_embed(Random_RM)

        # k = rearrange(k, 'b t (h d) -> b h t d', h=head)
        # q = rearrange(q, 'b t (h d) -> b h t d', h=head)
        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        attn_score = self.leakyrelu(attn_score)  # F.gelu

        ## ---1-----
        # attn_score = torch.matmul(torch.matmul(Random_RM, attn_score), Random_RM)
        ## ---2-----
        Lambda = self.randomatrix
        Lambda = Lambda.expand(b, -1, -1).to(device)
        attn_score = rearrange(attn_score, 'b h l t -> b h (l t)')
        attn_score = torch.einsum('blh,bhk->blk', [Lambda, attn_score])
        attn_score = rearrange(attn_score, 'b h (l k) -> b h l k', l=current_length)

        if self.layer_index < self.depth // 4:  # 4
            zero_vec = float('-inf') * torch.ones_like(attn_score)  # 将没有连接的边置为负无穷
            attn_score = torch.where(adj > 0, attn_score, zero_vec)
            # attn_score = attn_score.matmul(adj).matmul(attn_score).to(device)

        attn_score = F.softmax(attn_score, dim=-1)

        rep_adj = attn_score
        # if self.layer_index >= self.depth//4 & self.layer_index < self.depth//2:
        #     adj2order = torch.matmul(adj, adj) + adj
        #     zero_vec = 0 * torch.ones_like(attn_score)  # 将没有连接的边置为负无穷
        #     attn_score = torch.where(adj2order > 0, attn_score, zero_vec)

        # m_r = torch.ones_like(attn_score) * 0.1
        # attn_score = attn_score + torch.bernoulli(m_r)*-1e12

        # attn_score = self.attn_drop(attn_score)

        # rep_adj = similarity.matmul(rep_adj).matmul(similarity).to(device)
        # rep_adj = pointer_diag.matmul(rep_adj).matmul(pointer_diag).to(device)

        v = self.forward_conv_v(x, rep_adj)

        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=head)))

        out = self.proj_v(v)

        return out, rep_adj


class CrossAttention(nn.Module):
    def __init__(self,
                 depth,
                 i,
                 dim,
                 image_size,
                 patch_size,
                 heads=8,
                 dropout=0,
                 qkv_bias=False,
                 attn_drop=0.1,
                 proj_drop=0.,
                 downsample=0.,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=False
                 ):
        super().__init__()
        self.scale = dim ** -0.5
        # self.drop_ratio = 0.1

        # self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        # self.to_out = nn.Sequential(
        #     nn.Linear(dim, dim),
        #     nn.Dropout(dropout)
        # )

        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim
        self.num_heads = heads
        self.with_cls_token = with_cls_token

        self.length = int(image_size / patch_size) ** 2
        self.length2 = int(image_size / patch_size * downsample) ** 2

        self.layer_index = i
        self.depth = depth

        dim_in = dim
        dim_out = dim

        self.conv_proj_q = GCN(nfeat=dim_in,
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)
        self.conv_proj_k = GCN(nfeat=dim_in,
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)
        self.conv_proj_v = GCN(nfeat=dim_in,  # GCN_gru
                               head_num=heads,
                               nhid=dim_out,
                               image_size=image_size,
                               patch_size=patch_size,
                               stride=2,
                               padding=1,  # using 2 when kernel_size = 4
                               kernel_size=kernel_size,  # kernel_size of GCN
                               nclass=None,
                               dropout=dropout)

        # self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        # self.proj_q.weight.requires_grad = False
        # self.proj_k.weight.requires_grad = False
        # self.proj_v.weight.requires_grad = False

        # self.attn_drop = nn.Dropout(attn_drop)  #
        # self.proj_drop = nn.Dropout(proj_drop)  #
        self.leakyrelu = nn.LeakyReLU()

        # sparse_D = torch.ones((self.num_heads, self.length), requires_grad=True).to(device)
        # self.sparse_D = torch.nn.Parameter(sparse_D)
        # self.register_parameter("sparse_D", self.sparse_D)
        #
        # sparse_D2 = torch.ones((self.num_heads, self.length2), requires_grad=True).to(device)
        # self.sparse_D2 = torch.nn.Parameter(sparse_D2)
        # self.register_parameter("sparse_D2", self.sparse_D2)

        randomatrix = torch.randn((int(self.num_heads),
                                   int(self.num_heads)), requires_grad=True).to(device)
        self.randomatrix = torch.nn.Parameter(randomatrix)
        self.register_parameter("Ablah", self.randomatrix)

        # from torch.nn.parameter import Parameter
        # self.randomatrix = Parameter(torch.FloatTensor(int(self.num_heads), int(self.num_heads))).to(device)

        # self.proj_k_f = nn.Linear(2 * dim_in, self.num_heads, bias=qkv_bias)
        # self.proj_k_f2 = nn.Linear(self.num_heads, self.length, bias=qkv_bias)
        # self.proj_k_f3 = nn.Linear(self.num_heads, self.length2, bias=qkv_bias)
        # self.proj_k_f.weight.requires_grad = False
        # self.proj_k_f2.weight.requires_grad = False
        # self.proj_k_f3.weight.requires_grad = False

    def forward_conv_qk(self, x, resize_x, rep_adj, rep_adj_resize):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x, rep_adj)
            # q = F.dropout(F.relu(q), self.drop_ratio, training=self.training)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(resize_x, rep_adj_resize)
            # k = F.dropout(F.relu(k), self.drop_ratio, training=self.training)
        else:
            k = rearrange(resize_x, 'b c h w -> b (h w) c')

        return q, k

    def forward_conv_v(self, x, rep_adj):
        # if self.with_cls_token:
        #     cls_token, x = torch.split(x, [1, h * w], 1)

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x, rep_adj)
            # v = F.dropout(v, self.drop_ratio, training=self.training)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        return v

    def forward(self, x, resize_x, adj, adj_resize):
        x = rearrange(x, 'b t (h d) -> b h t d', h=self.num_heads)
        resize_x = rearrange(resize_x, 'b t (h d) -> b h t d', h=self.num_heads)
        current_length = adj.size(-1)
        b, head, num_rows, embedding_dim = x.size()

        # repeat_x = x.unsqueeze(3).cpu()
        # repeat_x_T = x.unsqueeze(2).cpu()
        # similarity = 1 - torch.cosine_similarity(repeat_x, repeat_x_T, dim=-1)
        # similarity = torch.softmax(similarity.to(device), dim=-1)
        # adj4qk = similarity.matmul(adj).matmul(similarity)

        # new_b = int(b*head)
        # matrices = rearrange(x, 'b h t d -> (b h) t d')
        # expanded_matrices = matrices.unsqueeze(1)  # Shape: (batch_size, 1, num_rows, embedding_dim)
        # replicated_matrices = expanded_matrices.repeat(1, num_rows, 1,1)  # Shape: (batch_size, num_rows, num_rows, embedding_dim)
        # flattened_replicated_matrices = replicated_matrices.view(-1, num_rows, embedding_dim)  # Shape: (batch_size * num_rows, num_rows, embedding_dim)
        # cosine_similarities = F.cosine_similarity(flattened_replicated_matrices, flattened_replicated_matrices, dim=2)
        # cosine_similarities = cosine_similarities.view(new_b, num_rows, num_rows)
        # cos_sim_matrix = rearrange(cosine_similarities, '(b h) t d->b h t d', h=self.num_heads)
        # adj4qk = cos_sim_matrix.matmul(adj).matmul(cos_sim_matrix)

        if (
                self.conv_proj_q is not None
                or self.conv_proj_k is not None
                or self.conv_proj_v is not None
        ):
            q, k = self.forward_conv_qk(x, resize_x, adj, adj_resize)

        attn_score = torch.einsum('bhlk,bhtk->bhlt', [q, k]) * self.scale

        attn_score = self.leakyrelu(attn_score)  # F.gelu

        ## ---1-----
        # attn_score = torch.matmul(torch.matmul(Random_RM, attn_score), Random_RM)
        ## ---2-----
        Lambda = self.randomatrix
        Lambda = Lambda.expand(b, -1, -1).to(device)
        attn_score = rearrange(attn_score, 'b h l t -> b h (l t)')
        attn_score = torch.einsum('blh,bhk->blk', [Lambda, attn_score])
        attn_score = rearrange(attn_score, 'b h (l k) -> b h l k', l=current_length)

        if self.layer_index < self.depth // 4:  # 4
            zero_vec = float('-inf') * torch.ones_like(attn_score)  # 将没有连接的边置为负无穷
            attn_score = torch.where(adj > 0, attn_score, zero_vec)
            # attn_score = attn_score.matmul(adj).matmul(attn_score).to(device)

        attn_score = F.softmax(attn_score, dim=-1)
        attn_score_resize = F.softmax(attn_score.transpose(2, 3), dim=-1)

        rep_adj = attn_score
        v = self.forward_conv_v(x, rep_adj)
        v = F.gelu((rearrange(v, 'b h t d -> b t (h d)', h=head)))
        out = self.proj_v(v)

        rep_adj_resize = attn_score_resize
        v_resize = self.forward_conv_v(resize_x, rep_adj_resize)
        v_resize = F.gelu((rearrange(v_resize, 'b h t d -> b t (h d)', h=head)))
        out_resize = self.proj_v(v_resize)

        return out, out_resize, rep_adj, rep_adj_resize


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 image_size,
                 patch_size,
                 kernel_size,
                 batch_size,
                 in_chans,
                 embed_dim,
                 stride,
                 padding,
                 norm_layer=None):
        super().__init__()
        # kernel_size = to_2tuple(kernel_size)
        # self.patch_size = patch_size

        # self.proj = ResNet18(embed_dim).to(device)

        # self.proj = nn.Conv2d(
        #     in_chans, embed_dim,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding
        # )

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_chans, int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # groups=in_chans
            )),
            #### ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size), int(image_size / patch_size)))),
            ('bn', nn.BatchNorm2d(int(embed_dim))),
            ('relu', nn.GELU()),
            ('conv2', nn.Conv2d(
                int(embed_dim), int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=int(embed_dim)
            )),
            ('pooling', nn.AdaptiveMaxPool2d((int(patch_size / 7), int(patch_size / 7)))),
            # ('bn', nn.BatchNorm2d(int(embed_dim))),
            # ('relu', nn.GELU()),
        ]))

        # self.proj = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(
        #         in_chans, int(embed_dim),
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         # groups=in_chans
        #     )),
        #     ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size / 4), int(image_size / patch_size / 4)))),
        #     ('bn', nn.BatchNorm2d(int(embed_dim))),
        #     ('relu', nn.GELU()),
        # ]))
        # self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        sp_features = self.proj(x).to(device)  # proj_conv  proj

        return sp_features


class ConvEmbed(nn.Module):
    """ Image to Conv Embedding

    """

    def __init__(self,
                 image_size,
                 patch_size,
                 kernel_size,
                 batch_size,
                 in_chans,
                 embed_dim,
                 stride,
                 padding,
                 norm_layer=None):
        super().__init__()
        # kernel_size = to_2tuple(kernel_size)
        # self.patch_size = patch_size

        # self.proj = ResNet18(embed_dim).to(device)

        # self.proj = nn.Conv2d(
        #     in_chans, embed_dim,
        #     kernel_size=kernel_size,
        #     stride=stride,
        #     padding=padding
        # )

        self.proj = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(
                in_chans, int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                # groups=in_chans
            )),
            #### ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size), int(image_size / patch_size)))),
            ('bn', nn.BatchNorm2d(int(embed_dim))),
            ('relu', nn.GELU()),
            ('conv2', nn.Conv2d(
                int(embed_dim), int(embed_dim),
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=int(embed_dim)
            )),
            ('pooling', nn.AdaptiveMaxPool2d((int(patch_size / 7), int(patch_size / 7)))),
            # ('bn', nn.BatchNorm2d(int(embed_dim))),
            # ('relu', nn.GELU()),
        ]))

        # self.proj = nn.Sequential(OrderedDict([
        #     ('conv1', nn.Conv2d(
        #         in_chans, int(embed_dim),
        #         kernel_size=kernel_size,
        #         stride=stride,
        #         padding=padding,
        #         # groups=in_chans
        #     )),
        #     ('pooling', nn.AdaptiveMaxPool2d((int(image_size / patch_size / 4), int(image_size / patch_size / 4)))),
        #     ('bn', nn.BatchNorm2d(int(embed_dim))),
        #     ('relu', nn.GELU()),
        # ]))
        # self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        sp_features = self.proj(x).to(device)  # proj_conv  proj

        return sp_features


class Transformer(nn.Module):
    def __init__(self, dim, depthin, depthout, heads, mlp_dim, dropout, image_size, patch_size, kernel_size, downsample,
                 batch_size, in_chans,
                 patch_stride, patch_padding, norm_layer=nn.LayerNorm):
        super().__init__()

        self.patch_embed = ConvEmbed(
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            batch_size=batch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=dim // 4,
            norm_layer=norm_layer
        )
        self.patch_dim = ((patch_size // 7) ** 2) * int(dim) // 4
        self.dim = dim
        self.patch_to_embedding = nn.Linear(self.patch_dim, dim).to(device)

        self.patch_embed_resize = ConvEmbed(
            image_size=image_size,
            patch_size=patch_size,
            kernel_size=kernel_size,
            batch_size=batch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=dim // 2,
            norm_layer=norm_layer
        )
        self.patch_dim_resize = ((patch_size // 7) ** 2) * int(dim) // 2
        self.dim = dim
        self.patch_to_embedding_resize = nn.Linear(self.patch_dim_resize, dim).to(device)  # //4

        self.layers = nn.ModuleList([])
        self.depthin4pool = depthin // 3  # 3
        for i in range(depthin):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim,
                                 Attention(depthin, i, dim, image_size=image_size, patch_size=patch_size, heads=heads,
                                           dropout=dropout, downsample=downsample, kernel_size=kernel_size))),
                # Residual(PreNorm(dim, FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout))),
                FeedForward(dim, mlp_dim, image_size, patch_size, dropout=dropout),
            ]))

        self.layers4bigpatch = nn.ModuleList([])
        self.depthout4pool = depthout // 3  # 3
        for i in range(depthout):
            self.layers4bigpatch.append(nn.ModuleList([
                Residual_out(PreNorm_out(int(dim), CrossAttention(depthout, i, int(dim), image_size=image_size,
                                                                  patch_size=patch_size, heads=heads,
                                                                  dropout=dropout, downsample=downsample,
                                                                  kernel_size=kernel_size))),
                FeedForward(int(dim), int(mlp_dim), image_size, patch_size, dropout=dropout),
            ]))

        self.dropout = nn.Dropout(dropout)

        # self.norm = nn.ModuleList([])
        # for _ in range(depth):
        #     self.norm.append(nn.LayerNorm(dim))

        self.patch_size = patch_size
        self.patchsalow = int(math.sqrt(image_size // patch_size))
        self.big_patch = image_size // self.patchsalow

        self.resize_transform = transforms.Resize((self.big_patch * 2, self.big_patch * 2))
        self.unloader = transforms.ToPILImage()
        self.tensor = transforms.ToTensor()
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.mean_tensor = torch.tensor(self.mean).view(3, 1, 1).to(device)
        self.std_tensor = torch.tensor(self.std).view(3, 1, 1).to(device)

        self.batch_size = batch_size
        self.head_num = heads
        UT = torch.randn((int(self.patchsalow * downsample) ** 2, dim), requires_grad=True).to(device)
        self.UT = torch.nn.Parameter(UT)
        self.register_parameter("Ablah2", self.UT)

        UT4bigpatch = torch.randn((int(self.patchsalow * downsample) ** 2, int(dim)), requires_grad=True).to(device)
        self.UT4bigpatch = torch.nn.Parameter(UT4bigpatch)
        self.register_parameter("Ablah3", self.UT4bigpatch)

        UT4bigpatch_resize = torch.randn((int(self.patchsalow * downsample) ** 2, int(dim)),
                                         requires_grad=True).to(device)
        self.UT4bigpatch_resize = torch.nn.Parameter(UT4bigpatch_resize)
        self.register_parameter("Ablah4", self.UT4bigpatch_resize)

        # self.Upool = nn.Sequential(
        #     nn.Linear(dim, int(image_size/patch_size*downsample)**2, bias=True),
        #     nn.Dropout(dropout)
        # )

        self.Upool_out = nn.Sequential(
            nn.Linear(dim, 1, bias=True),
        )
        self.Upool_out4bigpatch = nn.Sequential(
            nn.Linear(int(dim), 1, bias=True),
        )

    def forward(self, img, adj):
        p = self.big_patch
        b, n, imgh, imgw = img.shape

        x = rearrange(img, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p, p2=p)

        p = self.patch_size
        b_new, _, _, _ = x.shape
        x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p, p2=p)
        conv_img = self.patch_embed(x)
        conv_img = rearrange(conv_img, '(b s) c p1 p2 -> b s (c p1 p2)', b=b_new)

        x = self.patch_to_embedding(conv_img)

        imageors = []
        for i in range(b):
            original_tensor = img[i, :, :, :] * self.std_tensor + self.mean_tensor
            imageor = self.unloader(original_tensor).convert('RGB')
            imageor = self.resize_transform(imageor)
            img_resize = self.tensor(imageor)
            imageors.append(img_resize)

        imageors = torch.stack(imageors).to(device)
        img_resize = self.norm(imageors)

        # import matplotlib.pyplot as plt
        # plt.imshow(imageor, cmap="brg")
        # plt.show()
        # img_resize = self.resize_transform(imageors)

        resize_x = rearrange(img_resize, 'b c (h p1) (w p2) -> (b h w) (c) (p1) (p2)', p1=p * 2, p2=p * 2)
        resize_conv_img = self.patch_embed_resize(resize_x)
        resize_conv_img = rearrange(resize_conv_img, '(b s) c p1 p2 -> b s (c p1 p2)', b=b)
        resize_x = self.patch_to_embedding_resize(resize_conv_img)

        rep_adj = adj.expand(b_new, self.head_num, -1, -1).to(device)

        # global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        index = 0
        for attn, ff in self.layers:
            # x = attn(x, self.rep_adj, 0)
            if index < self.depthin4pool:
                x, rep_adj_new = attn(x, rep_adj, 0)
                rep_adj = rep_adj_new
                x = ff(x)
            else:
                if index == self.depthin4pool:
                    temp = torch.matmul(self.UT, x.permute(0, 2, 1))
                    # temp = self.Upool(x.permute(0, 2, 1))
                    # temp = F.gumbel_softmax(temp, dim=-1, tau=1.0)
                    temp = F.softmax(temp / 2, dim=-1)
                    x = torch.matmul(temp, x)

                    temp = temp.unsqueeze(dim=1).expand(b_new, self.head_num, -1, -1).to(device)
                    temp2 = torch.matmul(temp, rep_adj)
                    rep_adj = torch.matmul(temp2, temp.permute(0, 1, 3, 2))

                x, rep_adj_new = attn(x, rep_adj, 1)
                rep_adj = rep_adj_new
                x = ff(x)

            index = index + 1

        temp = self.Upool_out(x).permute(0, 2, 1)
        temp = F.softmax(temp / 2, dim=-1)
        x = torch.matmul(temp, x)

        x = rearrange(x, '(b s) c p -> b s (c p)', b=b)

        index = 0
        rep_adj = adj.expand(b, self.head_num, -1, -1).to(device)
        rep_adj_resize = adj.expand(b, self.head_num, -1, -1).to(device)
        for attn, ff in self.layers4bigpatch:  #
            if index < self.depthout4pool:
                x, resize_x, rep_adj_new, rep_adj_resize_new = attn(x, resize_x, rep_adj, rep_adj_resize)
                rep_adj = rep_adj_new
                rep_adj_resize = rep_adj_resize_new
                x = ff(x)
            else:
                if index == self.depthout4pool:
                    temp = torch.matmul(self.UT4bigpatch, x.permute(0, 2, 1))
                    # temp = self.Upool(x.permute(0, 2, 1))
                    # temp = F.gumbel_softmax(temp, dim=-1, tau=1.0)
                    temp = F.softmax(temp / 2, dim=-1)
                    x = torch.matmul(temp, x)
                    temp = temp.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
                    temp2 = torch.matmul(temp, rep_adj)
                    rep_adj = torch.matmul(temp2, temp.permute(0, 1, 3, 2))

                    temp = torch.matmul(self.UT4bigpatch_resize, resize_x.permute(0, 2, 1))
                    temp = F.softmax(temp / 2, dim=-1)
                    resize_x = torch.matmul(temp, resize_x)
                    temp = temp.unsqueeze(dim=1).expand(b, self.head_num, -1, -1).to(device)
                    temp2 = torch.matmul(temp, rep_adj_resize)
                    rep_adj_resize = torch.matmul(temp2, temp.permute(0, 1, 3, 2))

                x, resize_x, rep_adj_new, rep_adj_resize_new = attn(x, resize_x, rep_adj, rep_adj_resize)
                rep_adj = rep_adj_new
                rep_adj_resize = rep_adj_resize_new
                x = ff(x)

            index = index + 1

        temp = self.Upool_out4bigpatch(x).permute(0, 2, 1)
        temp = F.softmax(temp / 2, dim=-1)
        x_out = torch.matmul(temp, x)

        # temp = self.Upool_out4bigpatch(resize_x).permute(0, 2, 1)
        # temp = F.softmax(temp / 2, dim=-1)
        # x_out_resize = torch.matmul(temp, resize_x)
        # x_out = torch.cat([x_out, x_out_resize], dim=-1)

        return x_out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, kernel_size, downsample, batch_size, num_classes, dim, depthin,
                 depthout, heads,
                 mlp_dim, patch_stride, patch_pading, in_chans, dropout=0., emb_dropout=0., expansion_factor=1):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        pantchesalow = int(math.sqrt(image_size // patch_size))
        num_patches = pantchesalow ** 2

        adj_matrix = [[0 for i in range(num_patches)] for i in range(num_patches)]
        adj_matrix = torch.as_tensor(adj_matrix).float().to(device)

        for j in range(num_patches):
            if (j - pantchesalow - 1) >= 0:
                adj_matrix[j][j - 1] = 1
                adj_matrix[j][j - pantchesalow] = 1
                adj_matrix[j][j - pantchesalow - 1] = 1
                adj_matrix[j][j - pantchesalow + 1] = 1
            if (j + pantchesalow + 1) < num_patches:
                adj_matrix[j][j + 1] = 1
                adj_matrix[j][j + pantchesalow] = 1
                adj_matrix[j][j + pantchesalow - 1] = 1
                adj_matrix[j][j + pantchesalow + 1] = 1
        # adj_matrix = adj_matrix + torch.eye(adj_matrix.size(1), device=adj_matrix.device, dtype=adj_matrix.dtype)  # 为每个结点增加自环

        self.adj_matrix = adj_matrix

        self.patch_size = patch_size

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.patch_to_embedding = nn.Linear(patch_dim, dim)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depthin, depthout, heads, mlp_dim, dropout, image_size, patch_size,
                                       kernel_size, downsample,
                                       batch_size, in_chans, patch_stride=patch_stride, patch_padding=patch_pading)

        self.to_cls_token = nn.Identity()

        self.scale4dim = 1

        self.predictor = Predictor(dim, num_classes, self.scale4dim)

    def forward(self, img):
        x = self.transformer(img, self.adj_matrix)
        # x = self.to_cls_token(x[:, -1])
        pred = self.to_cls_token(x.squeeze())
        class_result = self.predictor(x.squeeze())
        return pred, class_result


# class MLP(nn.Module):
#     def __init__(self, dim, projection_size):
#         super().__init__()
#         hidden_size = dim*2
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_size),
#             nn.BatchNorm1d(hidden_size),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden_size, projection_size)
#         )
#
#     def forward(self, x):
#         return self.net(x)


class Predictor(nn.Module):
    def __init__(self, dim, num_classes, scale4dim):
        super().__init__()
        hidden_size = dim
        self.mlp_head = nn.Sequential(
            nn.Linear(int(dim * scale4dim), hidden_size),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        return self.mlp_head(x)  # .unsqueeze(0)
