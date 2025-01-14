import math
import random

import dgl
import dgl.function as fn
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch import EdgeWeightNorm
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import HeteroGraphConv
from torch import nn
from torch.nn import init


class FullAttention(nn.Module):

    def __init__(self, dim, scale=None, attention_dropout=0.0, qk_dim=32):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.q_fc = nn.Linear(dim, qk_dim)
        self.k_fc = nn.Linear(dim, qk_dim)
        self.v_fc = nn.Linear(dim, dim)
        self.att_weights = None

    def forward(self, queries, keys, values):
        N, K, D = keys.shape
        queries = self.q_fc(queries)  # (n,32)
        keys = self.k_fc(keys)  # (n,3,32)
        values = self.v_fc(values)  # (n,3,128)
        scale = self.scale or 1.0 / math.sqrt(D)
        queries = queries.unsqueeze(1)
        keys = keys.transpose(1, 2)
        scores = queries @ keys  # [n,1,k+1]
        A = scale * scores
        A = self.dropout(torch.softmax(A, dim=-1))
        V = A @ values
        return V.contiguous()


class SaConv(nn.Module):

    def __init__(self, in_feats, out_feats, k, qk_dim=32):
        super(SaConv, self).__init__()
        self._k = k
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.att = FullAttention(out_feats, qk_dim=qk_dim)

    def forward(self, graph, feat):
        raw_feat = feat.clone()

        def unnLaplacian(feat, D_invsqrt, graph):
            """High pass Feat * (I - D^-1/2 A D^-1/2) ==> L^sym"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        def A_tile(feat, D_invsqrt, graph):
            """Low pass Operation  ==> 2I - L^sym"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            L_sym = feat - graph.ndata.pop("h") * D_invsqrt
            return 2 * feat - L_sym

        with graph.local_scope():
            D_invsqrt = (torch.pow(graph.in_degrees().float().clamp(min=1),
                                   -0.5).unsqueeze(-1).to(feat.device))

            a_feat = A_tile(feat, D_invsqrt, graph)
            L_stack = [a_feat]  # low,highs

            for _ in range(self._k):
                feat = unnLaplacian(feat, D_invsqrt,
                                    graph)  # L^1·x + L^2·x + ··· +L^n·x [n,d]
                # h += self._theta[k]*feat
                L_stack.append(feat)  # K,n,d

        L_stack = torch.stack(L_stack, dim=0)
        L_stack = L_stack.transpose(0, 1)  # k,n,d-->n,k,d

        # 使用self-attention
        h = self.att(raw_feat, L_stack, L_stack)
        h = torch.sum(h, dim=1)

        return h


class SAGCN(nn.Module):

    def __init__(self, in_feats, h_feats, d=2, batch=False):
        super(SAGCN, self).__init__()

        self.conv = SaConv(h_feats, h_feats, d)
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h = self.conv(g, h)

        return h

    def testlarge(self, g, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h

    def batch(self, blocks, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)

        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            h0 = conv(blocks[0], h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h = self.act(h)
        h = self.linear4(h)
        return h
