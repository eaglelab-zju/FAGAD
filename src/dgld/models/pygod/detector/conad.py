# -*- coding: utf-8 -*-
"""Contrastive Attributed Network Anomaly Detection
with Data Augmentation (CONAD)"""
# Author: Zhiming Xu <zhimng.xu@gmail.com>
# License: BSD 2 clause
from copy import deepcopy

import numpy as np
import torch
from torch_geometric.nn import GCN
from torch_geometric.utils import dense_to_sparse

from . import DeepDetector
from ..nn import DOMINANTBase


class CONAD(DeepDetector):
    """
    Contrastive Attributed Network Anomaly Detection

    CONAD is an anomaly detector consisting of a shared graph
    convolutional encoder, a structure reconstruction decoder, and an
    attribute reconstruction decoder. The model is trained with both
    contrastive loss and structure/attribute reconstruction loss. The
    reconstruction mean square error of the decoders are defined as
    structure anomaly score and attribute anomaly score, respectively.

    See :cite:`xu2022contrastive` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``4``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    sigmoid_s : bool, optional
        Whether to use sigmoid function to scale the reconstructed
        structure. Default: ``False``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    weight : float, optional
        Weight between reconstruction of node feature and structure.
        Default: ``0.5``.
    eta : float, optional
        Weight between contrastive and reconstruction.Default: ``0.5``.
    margin : float, optional
        Margin in margin ranking loss. Default: ``0.5``.
    r : float, optional
        The rate of augmented anomalies. Default: ``0.2``.
    m : int, optional
        For densely connected nodes, the number of
        edges to add. Default: ``50``.
    k : int, optional
        Same as ``k`` in ``pygod.generator.gen_contextual_outlier``.
        Default: ``50``.
    f : int, optional
        For disproportionate nodes, the scale factor applied
        on their attribute value. Default: ``10``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs : optional
        Additional arguments for the backbone.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """

    def __init__(self,
                 hid_dim=64,
                 num_layers=4,
                 dropout=0.0,
                 weight_decay=0.0,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 weight=0.5,
                 eta=0.5,
                 margin=0.5,
                 r=0.2,
                 m=50,
                 k=50,
                 f=10,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(CONAD, self).__init__(hid_dim=hid_dim,
                                    num_layers=num_layers,
                                    dropout=dropout,
                                    weight_decay=weight_decay,
                                    act=act,
                                    backbone=backbone,
                                    contamination=contamination,
                                    lr=lr,
                                    epoch=epoch,
                                    gpu=gpu,
                                    batch_size=batch_size,
                                    num_neigh=num_neigh,
                                    verbose=verbose,
                                    save_emb=save_emb,
                                    compile_model=compile_model,
                                    **kwargs)

        # model param
        self.weight = weight
        self.sigmoid_s = sigmoid_s
        self.eta = eta

        # other param
        self.r = r
        self.m = m
        self.k = k
        self.f = f

        self.margin_loss_func = torch.nn.MarginRankingLoss(margin=margin)

    def process_graph(self, data):
        DOMINANTBase.process_graph(data)

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes, self.hid_dim)
        return DOMINANTBase(in_dim=self.in_dim,
                            hid_dim=self.hid_dim,
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            act=self.act,
                            sigmoid_s=self.sigmoid_s,
                            backbone=self.backbone,
                            **kwargs).to(self.device)

    def forward_model(self, data):
        batch_size = data.batch_size
        node_idx = data.n_id

        x = data.x.to(self.device)
        s = data.s.to(self.device)
        edge_index = data.edge_index.to(self.device)
        if self.model.training:
            x_aug, edge_index_aug, label_aug = self._data_augmentation(data)
            x_aug = x_aug.to(self.device)
            edge_index_aug = edge_index_aug.to(self.device)
            label_aug = label_aug.to(self.device)

            _, _ = self.model(x_aug, edge_index_aug)
            h_aug = self.model.emb

        x_, s_ = self.model(x, edge_index)
        h = self.model.emb

        score = self.model.loss_func(
            x[:batch_size],
            x_[:batch_size],
            s[:batch_size, node_idx],
            s_[:batch_size],
            self.weight,
        )

        if self.model.training:
            margin_loss = self.margin_loss_func(h, h, h_aug) * label_aug
            loss = self.eta * torch.mean(score) + (
                1 - self.eta) * torch.mean(margin_loss)
        else:
            loss = torch.mean(score)

        return loss, score.detach().cpu()

    def _data_augmentation(self, data):
        """
        Data augmentation on the input graph. Four types of
        pseudo anomalies will be injected:
            Attribute, deviated
            Attribute, disproportionate
            Structure, high-degree
            Structure, outlying

        Parameters
        -----------
        data : torch_geometric.data.Data
            The input graph data.

        Returns
        -------
        feat_aug, adj_aug, label_aug : augmented
            attribute matrix, adjacency matrix, and
            pseudo anomaly label to train contrastive
            graph representations
        """
        rate = self.r
        num_added_edge = self.m
        surround = self.k
        scale_factor = self.f

        x = data.x
        adj = data.s
        node_idx = data.n_id

        batch_size = adj.shape[0]
        num_nodes = adj.shape[1]

        adj_aug, feat_aug = deepcopy(adj), deepcopy(x)
        label_aug = torch.zeros(batch_size, dtype=torch.int32)

        prob = torch.rand(batch_size)
        label_aug[prob < rate] = 1

        # high-degree
        hd_mask = prob < rate / 4
        n_hd = torch.sum(hd_mask)
        edges_mask = torch.rand(n_hd, num_nodes) < num_added_edge / num_nodes
        edges_mask = edges_mask
        adj_aug[hd_mask, :] = edges_mask.float()

        # outlying
        ol_mask = torch.logical_and(rate / 4 <= prob, prob < rate / 2)
        adj_aug[ol_mask, :] = 0

        # deviated
        dv_mask = torch.logical_and(rate / 2 <= prob, prob < rate * 3 / 4)
        feat_c = feat_aug[torch.randperm(batch_size)[:surround]]
        ds = torch.cdist(feat_aug[dv_mask], feat_c)
        feat_aug[dv_mask] = feat_c[torch.argmax(ds, 1)]

        # disproportionate
        mul_mask = torch.logical_and(rate * 3 / 4 <= prob, prob < rate * 7 / 8)
        div_mask = rate * 7 / 8 <= prob
        feat_aug[mul_mask] *= scale_factor
        feat_aug[div_mask] /= scale_factor

        edge_index_aug = dense_to_sparse(adj_aug)[0]
        inv_idx = torch.zeros(num_nodes, dtype=torch.int64)
        inv_idx[node_idx] = torch.arange(batch_size)
        edge_index_aug[1] = inv_idx[edge_index_aug[1]]

        return feat_aug, edge_index_aug, label_aug
