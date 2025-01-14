import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
from torch import nn

from .SAGCN import SAGCN


class Encoder(nn.Module):

    def __init__(self,
                 in_feats,
                 hid_feats=128,
                 mlp_hids=[512, 256],
                 k=2,
                 struct_dec_act=None):
        super().__init__()

        self.encoder = SAGCN(in_feats, hid_feats, k)
        self.projection = MLPHead(hid_feats, mlp_hids[0], mlp_hids[1])
        self.attr_decoder = SAGCN(mlp_hids[1], in_feats, k)
        self.struct_decoder = lambda h: h @ h.T
        self.struct_dec_act = struct_dec_act

    def forward(self, g, x):
        x = self.encoder(g, x)
        x = self.projection(x)

        x_hat = self.attr_decoder(g, x)
        # reconstruct adjacency matrix
        a_hat = self.struct_decoder(x)
        if self.struct_dec_act is not None:
            a_hat = self.struct_dec_act(a_hat)

        return x_hat, a_hat


class MLPHead(nn.Module):

    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size),
        )

    def forward(self, x):
        h = self.net(x)
        return h


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout: float = 0.0, act=lambda x: x):
        super().__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class LowConv(nn.Module):

    def __init__(self, act=None):
        super().__init__()
        self.act = act

    def forward(self, graph, feat):
        h = feat

        def A_tile(feat, D_invsqrt, graph):
            """Low pass Operation  ==> 2I - L^sym"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            L_sym = feat - graph.ndata.pop("h") * D_invsqrt
            return 2 * feat - L_sym

        with graph.local_scope():
            D_invsqrt = (torch.pow(graph.in_degrees().float().clamp(min=1),
                                   -0.5).unsqueeze(-1).to(feat.device))

            h = A_tile(h, D_invsqrt, graph)
            # h = A_tile(h, D_invsqrt, graph)

        if self.act != None:
            h = self.act(h)
        return h


class LowGCN(nn.Module):

    def __init__(self,
                 in_size,
                 hid_size_list,
                 mlp_hids_list=[256, 128],
                 act=None):
        super().__init__()
        self.hid_size_list = hid_size_list

        if len(hid_size_list) == 1:
            # input layer
            self.add_module("linear0",
                            nn.Linear(in_size, hid_size_list[0], bias=False))
            self.add_module("conv0", LowConv(act=None))
        else:
            # input layer
            self.add_module("linear0",
                            nn.Linear(in_size, hid_size_list[0], bias=False))
            self.add_module("conv0", LowConv(act=act))

            # hidden layer
            for i, d in enumerate(hid_size_list[1:-1]):
                self.add_module(f"linear{i+1}",
                                nn.Linear(hid_size_list[i], d, bias=False))
                self.add_module(f"conv{i+1}", LowConv(act=act))

            # output layer
            self.add_module(
                f"linear{len(hid_size_list)-1}",
                nn.Linear(hid_size_list[-2], hid_size_list[-1], bias=False),
            )
            self.add_module(f"conv{len(hid_size_list)-1}", LowConv(act=None))

        self.add_module(
            "projection",
            MLPHead(hid_size_list[-1], mlp_hids_list[0], mlp_hids_list[1]))

    def forward(self, graph, feat):
        h = feat
        for i in range(len(self.hid_size_list)):
            h = eval(f"self.linear{i}")(h)
            h = eval(f"self.conv{i}")(graph, h)
        h = self.projection(h)
        return h


class HighConv(nn.Module):

    def __init__(self, act=None):
        super().__init__()
        self.act = act

    def forward(self, graph, feat):
        h = feat

        def unnLaplacian(feat, D_invsqrt, graph):
            """High pass Feat * (I - D^-1/2 A D^-1/2) ==> L^sym"""
            graph.ndata["h"] = feat * D_invsqrt
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            return feat - graph.ndata.pop("h") * D_invsqrt

        with graph.local_scope():
            D_invsqrt = (torch.pow(graph.in_degrees().float().clamp(min=1),
                                   -0.5).unsqueeze(-1).to(feat.device))

            h = unnLaplacian(h, D_invsqrt, graph)
            # h = unnLaplacian(h, D_invsqrt, graph)
            # h = unnLaplacian(h, D_invsqrt, graph)

        if self.act != None:
            h = self.act(h)

        return h


class HighGCN(nn.Module):

    def __init__(self,
                 in_size,
                 hid_size_list,
                 mlp_hids_list=[256, 128],
                 act=None):
        super().__init__()
        self.hid_size_list = hid_size_list

        if len(hid_size_list) == 1:
            # input layer
            self.add_module("linear0",
                            nn.Linear(in_size, hid_size_list[0], bias=False))
            self.add_module("conv0", HighConv(act=None))
        else:
            # input layer
            self.add_module("linear0",
                            nn.Linear(in_size, hid_size_list[0], bias=False))
            self.add_module("conv0", HighConv(act=act))

            # hidden layer
            for i, d in enumerate(hid_size_list[1:-1]):
                self.add_module(f"linear{i+1}",
                                nn.Linear(hid_size_list[i], d, bias=False))
                self.add_module(f"conv{i+1}", HighConv(act=act))

            # output layer
            self.add_module(
                f"linear{len(hid_size_list)-1}",
                nn.Linear(hid_size_list[-2], hid_size_list[-1], bias=False),
            )
            self.add_module(f"conv{len(hid_size_list)-1}", HighConv(act=None))

        self.add_module(
            "projection",
            MLPHead(hid_size_list[-1], mlp_hids_list[0], mlp_hids_list[1]))

    def forward(self, graph, feat):
        h = feat
        for i in range(len(self.hid_size_list)):
            h = eval(f"self.linear{i}")(h)
            h = eval(f"self.conv{i}")(graph, h)
        h = self.projection(h)
        return h


class GCN(nn.Module):

    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation=None,
                 dropout=0.0):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, bias=False, activation=activation))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(
                GraphConv(n_hidden,
                          n_hidden,
                          bias=False,
                          activation=activation))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes, bias=False))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h)
        return h
