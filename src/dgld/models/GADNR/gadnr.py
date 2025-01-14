import argparse
import math
import random
import statistics
import sys

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.optimize
import seaborn as sb
import sklearn as sk
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import CitationGraphDataset
from dgld.models.pygod.metric import eval_roc_auc
from dgld.models.pygod.utils import load_data
from dgld.models.pygod.utils.utility import check_parameter
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from the_utils import set_seed
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import GraphSAGE
from torch_geometric.nn import PNAConv
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import normalize_features
from torch_geometric.utils import add_self_loops
from tqdm import tqdm

# from dgl.nn import GINConv, GraphConv, SAGEConv
# from dgld.models.pygod.generator import gen_contextual_outliers, gen_structural_outliers


def _normalize(x):
    x_min = x.min()
    x_max = x.max()
    x_norm = (x - x_min) / x_max
    return x_norm


def gen_joint_structural_outliers(data, m, n, random_state=None):
    """
    We randomly select n nodes from the network which will be the anomalies
    and for each node we select m nodes from the network.
    We connect each of n nodes with the m other nodes.

    Parameters
    ----------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The input data.
    m : int
        Number nodes in the outlier cliques.
    n : int
        Number of outlier cliques.
    p : int, optional
        Probability of edge drop in cliques. Default: ``0``.
    random_state : int, optional
        The seed to control the randomness, Default: ``None``.

    Returns
    -------
    data : PyTorch Geometric Data instance (torch_geometric.data.Data)
        The structural outlier graph with injected edges.
    y_outlier : torch.Tensor
        The outlier label tensor where 1 represents outliers and 0 represents
        regular nodes.
    """

    if not isinstance(data, Data):
        raise TypeError("data should be torch_geometric.data.Data")

    if isinstance(m, int):
        check_parameter(m, low=0, high=data.num_nodes, param_name="m")
    else:
        raise ValueError("m should be int, got %s" % m)

    if isinstance(n, int):
        check_parameter(n, low=0, high=data.num_nodes, param_name="n")
    else:
        raise ValueError("n should be int, got %s" % n)

    check_parameter(m * n, low=0, high=data.num_nodes, param_name="m*n")

    if random_state:
        np.random.seed(random_state)

    outlier_idx = np.random.choice(data.num_nodes, size=n, replace=False)
    all_nodes = [i for i in range(data.num_nodes)]
    rem_nodes = []

    for node in all_nodes:
        if node is not outlier_idx:
            rem_nodes.append(node)

    new_edges = []

    # connect all m nodes in each clique
    for i in range(0, n):
        other_idx = np.random.choice(data.num_nodes, size=m, replace=False)
        for j in other_idx:
            new_edges.append(torch.tensor([[i, j]], dtype=torch.long))

    new_edges = torch.cat(new_edges)

    y_outlier = torch.zeros(data.x.shape[0], dtype=torch.long)
    y_outlier[outlier_idx] = 1

    data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    return data, y_outlier


def KL_neighbor_loss(predictions, targets, mask_len):
    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]
    h_dim = x1.shape[1]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / max(
        (nn - 1), 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / max(
        (nn - 1), 1)

    eye = torch.eye(h_dim)
    cov_x1 = cov_x1 + eye
    cov_x2 = cov_x2 + eye

    KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim +
                     torch.trace(torch.inverse(cov_x2).matmul(cov_x1)) +
                     (mean_x2 - mean_x1).reshape(1, -1).matmul(
                         torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))
    KL_loss = KL_loss.to(device)
    return KL_loss


def W2_neighbor_loss(predictions, targets, mask_len):

    x1 = predictions.squeeze().cpu().detach()
    x2 = targets.squeeze().cpu().detach()

    mean_x1 = x1.mean(0)
    mean_x2 = x2.mean(0)

    nn = x1.shape[0]

    cov_x1 = (x1 - mean_x1).transpose(1, 0).matmul(x1 - mean_x1) / (nn - 1)
    cov_x2 = (x2 - mean_x2).transpose(1, 0).matmul(x2 - mean_x2) / (nn - 1)

    W2_loss = torch.square(mean_x1 - mean_x2).sum() + torch.trace(
        cov_x1 + cov_x2 +
        2 * sqrtm(sqrtm(cov_x1) @ (cov_x2.numpy()) @ (sqrtm(cov_x1))))

    return W2_loss


class MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = self.linears[layer](h)

                if len(h.shape) > 2:
                    h = torch.transpose(h, 0, 1)
                    h = torch.transpose(h, 1, 2)

                h = self.batch_norms[layer](h)

                if len(h.shape) > 2:
                    h = torch.transpose(h, 1, 2)
                    h = torch.transpose(h, 0, 1)

                h = F.relu(h)
                # h = F.relu(self.linears[layer](h))

            return self.linears[self.num_layers - 1](h)


class MLP_generator(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(MLP_generator, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)
        self.linear3 = nn.Linear(output_dim, output_dim)
        self.linear4 = nn.Linear(output_dim, output_dim)

    def forward(self, embedding):
        neighbor_embedding = F.relu(self.linear(embedding))
        neighbor_embedding = F.relu(self.linear2(neighbor_embedding))
        neighbor_embedding = F.relu(self.linear3(neighbor_embedding))
        neighbor_embedding = self.linear4(neighbor_embedding)
        return neighbor_embedding


class PairNorm(nn.Module):

    def __init__(self, mode="PN", scale=10):

        assert mode in ["None", "PN", "PN-SI", "PN-SCS"]
        super(PairNorm, self).__init__()
        self.mode = mode
        self.scale = scale

        # Scale can be set based on origina data, and also the current feature lengths.
        # We leave the experiments to future. A good pool we used for choosing scale:
        # [0.1, 1, 10, 50, 100]

    def forward(self, x):
        if self.mode == "None":
            return x
        col_mean = x.mean(dim=0)
        if self.mode == "PN":
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean
        if self.mode == "PN-SI":
            x = x - col_mean
            rownorm_individual = (1e-6 +
                                  x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual
        if self.mode == "PN-SCS":
            rownorm_individual = (1e-6 +
                                  x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean
        return x


# FNN
class FNN(nn.Module):

    def __init__(self, in_features, hidden, out_features, layer_num):
        super(FNN, self).__init__()
        self.linear1 = MLP(layer_num, in_features, hidden, out_features)
        self.linear2 = nn.Linear(out_features, out_features)

    def forward(self, embedding):
        x = self.linear1(embedding)
        x = self.linear2(F.relu(x))
        x = F.relu(x)
        return x


def evaluate(model, embeddings, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(embeddings)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


# generate ground truth neighbors Hv
def generate_gt_neighbor(neighbor_dict, node_embeddings, neighbor_num_list,
                         in_dim):
    max_neighbor_num = max(neighbor_num_list)
    all_gt_neighbor_embeddings = []
    for i, embedding in enumerate(node_embeddings):
        neighbor_indexes = neighbor_dict[i]
        neighbor_embeddings = []
        for index in neighbor_indexes:
            neighbor_embeddings.append(node_embeddings[index].tolist())
        if len(neighbor_embeddings) < max_neighbor_num:
            for _ in range(max_neighbor_num - len(neighbor_embeddings)):
                neighbor_embeddings.append(torch.zeros(in_dim).tolist())
        all_gt_neighbor_embeddings.append(neighbor_embeddings)
    return all_gt_neighbor_embeddings


# Main Autoencoder structure here
class GNNStructEncoder(nn.Module):

    def __init__(
        self,
        in_dim0,
        in_dim,
        hidden_dim,
        layer_num,
        sample_size,
        device,
        neighbor_num_list,
        GNN_name="GIN",
        norm_mode="PN-SCS",
        norm_scale=20,
        lambda_loss1=0.01,
        lambda_loss2=0.001,
        lambda_loss3=0.0001,
    ):

        super(GNNStructEncoder, self).__init__()

        self.mlp0 = nn.Linear(in_dim0, hidden_dim)
        self.norm = PairNorm(norm_mode, norm_scale)
        self.out_dim = hidden_dim
        self.lambda_loss1 = lambda_loss1
        self.lambda_loss2 = lambda_loss2
        self.lambda_loss3 = lambda_loss3
        # GNN Encoder
        if GNN_name == "GIN":
            self.linear1 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv1 = GINConv(self.linear1)
            self.linear2 = MLP(layer_num, hidden_dim, hidden_dim, hidden_dim)
            self.graphconv2 = GINConv(self.linear2)
        elif GNN_name == "GCN":
            self.graphconv1 = GCNConv(hidden_dim, hidden_dim)
            self.graphconv2 = GCNConv(hidden_dim, hidden_dim)
        elif GNN_name == "GAT":
            self.graphconv1 = GATConv(hidden_dim, hidden_dim)
            self.graphconv2 = GATConv(hidden_dim, hidden_dim)
        else:
            self.graphconv1 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
            # self.graphconv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')

            # self.graphconv1 = GraphSAGE(hidden_dim, hidden_dim, aggr='mean', num_layers=1)

        self.neighbor_num_list = neighbor_num_list
        self.tot_node = len(neighbor_num_list)

        self.gaussian_mean = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(
                -0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(sample_size, hidden_dim).uniform_(
                -0.5 / hidden_dim, 0.5 / hidden_dim)).to(device)
        self.m = torch.distributions.Normal(
            torch.zeros(sample_size, hidden_dim),
            torch.ones(sample_size, hidden_dim))

        self.m_batched = torch.distributions.Normal(
            torch.zeros(sample_size, self.tot_node, hidden_dim),
            torch.ones(sample_size, self.tot_node, hidden_dim),
        )

        self.m_h = torch.distributions.Normal(
            torch.zeros(sample_size, hidden_dim),
            50 * torch.ones(sample_size, hidden_dim),
        )

        # Before MLP Gaussian Means, and std

        self.mlp_gaussian_mean = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 /
                                                   hidden_dim)).to(device)
        self.mlp_gaussian_log_sigma = nn.Parameter(
            torch.FloatTensor(hidden_dim).uniform_(-0.5 / hidden_dim, 0.5 /
                                                   hidden_dim)).to(device)
        self.mlp_m = torch.distributions.Normal(torch.zeros(hidden_dim),
                                                torch.ones(hidden_dim))

        self.mlp_mean = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.mlp_sigma = FNN(hidden_dim, hidden_dim, hidden_dim, 3)
        self.softplus = nn.Softplus()

        # self.mean_agg = SAGEConv(hidden_dim, hidden_dim, aggr='mean', normalize = True)
        self.mean_agg = GraphSAGE(hidden_dim,
                                  hidden_dim,
                                  aggr="mean",
                                  num_layers=1)
        self.std_agg = PNAConv(
            hidden_dim,
            hidden_dim,
            aggregators=["std"],
            scalers=["identity"],
            deg=neighbor_num_list,
        )
        self.layer1_generator = MLP_generator(hidden_dim, hidden_dim)

        # Decoders
        self.degree_decoder = FNN(hidden_dim, hidden_dim, 1, 4)
        self.feature_decoder = FNN(hidden_dim, hidden_dim, in_dim, 3)
        self.degree_loss_func = nn.MSELoss()
        self.feature_loss_func = nn.MSELoss()
        self.pool = mp.Pool(4)
        self.in_dim = in_dim
        self.sample_size = sample_size
        self.init_projection = FNN(in_dim, hidden_dim, hidden_dim, 1)

    def forward_encoder(self, x, edge_index):

        # Apply graph convolution and activation, pair-norm to avoid trivial solution
        h0 = self.mlp0(x)
        l1 = self.graphconv1(h0, edge_index)
        return l1, h0

    # Sample neighbors from neighbor set, if the length of neighbor set less than sample size, then do the padding.
    def sample_neighbors(self, indexes, neighbor_dict, gt_embeddings):
        sampled_embeddings_list = []
        mark_len_list = []
        for index in indexes:
            sampled_embeddings = []
            neighbor_indexes = neighbor_dict[index]
            if len(neighbor_indexes) < self.sample_size:
                mask_len = len(neighbor_indexes)
                sample_indexes = neighbor_indexes
            else:
                sample_indexes = random.sample(neighbor_indexes,
                                               self.sample_size)
                mask_len = self.sample_size
            for index in sample_indexes:
                sampled_embeddings.append(gt_embeddings[index].tolist())
            if len(sampled_embeddings) < self.sample_size:
                for _ in range(self.sample_size - len(sampled_embeddings)):
                    sampled_embeddings.append(
                        torch.zeros(self.out_dim).tolist())
            sampled_embeddings_list.append(sampled_embeddings)
            mark_len_list.append(mask_len)

        return sampled_embeddings_list, mark_len_list

    def reconstruction_neighbors2(self, l1, h0, edge_index):

        recon_loss = 0
        recon_loss_per_node = []

        sample_sz_per_node = [self.sample_size] * self.tot_node

        # mean_neigh = self.graphconv1(h0, edge_index)
        # mean_neigh = self.graphconv2(mean_neigh, edge_index)
        # mean_neigh = l1
        mean_neigh = self.mean_agg(h0, edge_index).detach()
        std_neigh = self.std_agg(h0, edge_index).detach()

        cov_neigh = torch.bmm(std_neigh.unsqueeze(dim=-1),
                              std_neigh.unsqueeze(dim=1))

        target_mean = mean_neigh
        target_cov = cov_neigh

        self_embedding = l1
        # self_embedding = _normalize(self_embedding)

        self_embedding = self_embedding.unsqueeze(0)
        self_embedding = self_embedding.repeat(self.sample_size, 1, 1)
        generated_mean = self.mlp_mean(self_embedding)
        generated_sigma = self.mlp_sigma(self_embedding)

        std_z = self.m_batched.sample().to(device)
        var = generated_mean + generated_sigma.exp() * std_z
        nhij = self.layer1_generator(var)

        generated_mean = torch.mean(nhij, dim=0)
        generated_std = torch.std(nhij, dim=0)
        generated_cov = (torch.bmm(generated_std.unsqueeze(dim=-1),
                                   generated_std.unsqueeze(dim=1)) /
                         self.sample_size)

        tot_nodes = l1.shape[0]
        h_dim = l1.shape[1]

        single_eye = torch.eye(h_dim).to(device)
        single_eye = single_eye.unsqueeze(dim=0)
        batch_eye = single_eye.repeat(tot_nodes, 1, 1)

        target_cov = target_cov + batch_eye
        generated_cov = generated_cov + batch_eye

        det_target_cov = torch.linalg.det(target_cov)
        det_generated_cov = torch.linalg.det(generated_cov)
        trace_mat = torch.matmul(torch.inverse(generated_cov), target_cov)

        x = torch.bmm(
            torch.unsqueeze(generated_mean - target_mean, dim=1),
            torch.inverse(generated_cov),
        )
        y = torch.unsqueeze(generated_mean - target_mean, dim=-1)
        z = torch.bmm(x, y).squeeze()

        KL_loss = 0.5 * (
            torch.log(det_target_cov / det_generated_cov) - h_dim +
            trace_mat.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1) + z)

        recon_loss = torch.mean(KL_loss)
        recon_loss_per_node = KL_loss

        return recon_loss, recon_loss_per_node

        # self_embedding_min = self_embedding.min()
        # self_embedding_max = self_embedding.max()
        # self_embedding = (self_embedding - self_embedding_min)/(self_embedding_max+1e-14)
        # generated_mean = self.mean_generator(self_embedding)
        # generated_mean = _normalize(generated_mean)
        # generated_std = self.std_generator(self_embedding)
        # generated_std = _normalize(generated_std)
        # generated_std = generated_std.abs().squeeze()
        # generated_std = self.softplus(generated_std)
        # generated_std = generated_std.exp().squeeze()

        # cov_x1 = (x1-mean_x1).transpose(1,0).matmul(x1-mean_x1) / max((nn-1),1)
        # cov_x2 = (x2-mean_x2).transpose(1,0).matmul(x2-mean_x2) / max((nn-1),1)

        # generated_cov = generated_cov + torch.rand(generated_cov.shape).to(device)
        # generated_cov = generated_cov * batch_eye

        # print(torch.isfinite(det_target_cov).all())
        # print(torch.isfinite(det_generated_cov).all())

        # KL_loss = 0.5 * (math.log(torch.det(cov_x1) / torch.det(cov_x2)) - h_dim  + torch.trace(torch.inverse(cov_x2).matmul(cov_x1))
        #     + (mean_x2 - mean_x1).reshape(1,-1).matmul(torch.inverse(cov_x2)).matmul(mean_x2 - mean_x1))

    def reconstruction_neighbors(
        self,
        FNN_generator,
        neighbor_indexes,
        neighbor_dict,
        from_layer,
        to_layer,
        device,
    ):

        local_index_loss = 0
        local_index_loss_per_node = []
        sampled_embeddings_list, mark_len_list = self.sample_neighbors(
            neighbor_indexes, neighbor_dict, to_layer)
        for i, neighbor_embeddings1 in enumerate(sampled_embeddings_list):
            # Generating h^k_v, reparameterization trick

            # print(len(neighbor_embeddings1))

            index = neighbor_indexes[i]
            mask_len1 = mark_len_list[i]
            mean = from_layer[index].repeat(self.sample_size, 1)
            mean = self.mlp_mean(mean)
            sigma = from_layer[index].repeat(self.sample_size, 1)
            sigma = self.mlp_sigma(sigma)
            std_z = self.m.sample().to(device)
            var = mean + sigma.exp() * std_z
            nhij = FNN_generator(var)

            generated_neighbors = nhij
            sum_neighbor_norm = 0

            for indexi, generated_neighbor in enumerate(generated_neighbors):
                sum_neighbor_norm += torch.norm(
                    generated_neighbor) / math.sqrt(self.out_dim)
            generated_neighbors = torch.unsqueeze(generated_neighbors,
                                                  dim=0).to(device)
            target_neighbors = torch.unsqueeze(
                torch.FloatTensor(neighbor_embeddings1), dim=0).to(device)

            if args.neigh_loss == "KL":
                KL_loss = KL_neighbor_loss(generated_neighbors,
                                           target_neighbors, mask_len1)
                local_index_loss += KL_loss
                local_index_loss_per_node.append(KL_loss)

            else:
                W2_loss = W2_neighbor_loss(generated_neighbors,
                                           target_neighbors, mask_len1)
                local_index_loss += W2_loss
                local_index_loss_per_node.append(W2_loss)

        local_index_loss_per_node = torch.stack(local_index_loss_per_node)
        return local_index_loss, local_index_loss_per_node

    def neighbor_decoder(self, gij, ground_truth_degree_matrix, h0,
                         neighbor_dict, device, h, edge_index):

        # Degree decoder below:
        tot_nodes = gij.shape[0]
        degree_logits = self.degree_decoding(gij)
        ground_truth_degree_matrix = torch.unsqueeze(
            ground_truth_degree_matrix, dim=1)
        degree_loss = self.degree_loss_func(degree_logits,
                                            ground_truth_degree_matrix.float())
        degree_loss_per_node = (degree_logits -
                                ground_truth_degree_matrix).pow(2)
        _, degree_masks = torch.max(degree_logits.data, dim=1)
        h_loss = 0
        feature_loss = 0
        # layer 1
        loss_list = []
        loss_list_per_node = []
        feature_loss_list = []
        # Sample multiple times to remove noise
        for _ in range(3):
            local_index_loss_sum = 0
            local_index_loss_sum_per_node = []
            indexes = []
            h0_prime = self.feature_decoder(gij)
            feature_losses = self.feature_loss_func(h0, h0_prime)
            feature_losses_per_node = (h0 - h0_prime).pow(2).mean(1)
            feature_loss_list.append(feature_losses_per_node)

            # local_index_loss, local_index_loss_per_node = self.reconstruction_neighbors(self.layer1_generator, indexes, neighbor_dict, gij, h0, device)
            local_index_loss, local_index_loss_per_node = (
                self.reconstruction_neighbors2(gij, h0, edge_index))

            # print(local_index_loss, local_index_loss2)
            # print(local_index_loss_per_node, local_index_loss_per_node2)

            loss_list.append(local_index_loss)
            loss_list_per_node.append(local_index_loss_per_node)

        loss_list = torch.stack(loss_list)
        h_loss += torch.mean(loss_list)

        loss_list_per_node = torch.stack(loss_list_per_node)
        h_loss_per_node = torch.mean(loss_list_per_node, dim=0)

        feature_loss_per_node = torch.mean(torch.stack(feature_loss_list),
                                           dim=0)
        feature_loss += torch.mean(torch.stack(feature_loss_list))

        h_loss_per_node = h_loss_per_node.reshape(tot_nodes, 1)
        degree_loss_per_node = degree_loss_per_node.reshape(tot_nodes, 1)
        feature_loss_per_node = feature_loss_per_node.reshape(tot_nodes, 1)

        loss = (self.lambda_loss1 * h_loss + degree_loss * self.lambda_loss3 +
                self.lambda_loss2 * feature_loss)
        loss_per_node = (self.lambda_loss1 * h_loss_per_node +
                         degree_loss_per_node * self.lambda_loss3 +
                         self.lambda_loss2 * feature_loss_per_node)

        return (
            loss,
            loss_per_node,
            h_loss_per_node,
            degree_loss_per_node,
            feature_loss_per_node,
        )

    def degree_decoding(self, node_embeddings):
        degree_logits = F.relu(self.degree_decoder(node_embeddings))
        return degree_logits

    def forward(self, edge_index, x, ground_truth_degree_matrix, neighbor_dict,
                device):

        # Generate GNN encodings
        l1, h0 = self.forward_encoder(x, edge_index)
        loss, loss_per_node, h_loss, degree_loss, feature_loss = self.neighbor_decoder(
            l1, ground_truth_degree_matrix, h0, neighbor_dict, device, x,
            edge_index)

        return loss, loss_per_node, h_loss, degree_loss, feature_loss


# Training
def train(
    data,
    y,
    lr,
    epoch,
    device,
    encoder,
    lambda_loss1,
    lambda_loss2,
    lambda_loss3,
    hidden_dim,
    sample_size=10,
    loss_step=20,
    real_loss=False,
    calculate_contextual=False,
    calculate_structural=False,
):
    """
    Main training function
    INPUT:
    -----------------------
    data : torch geometric dataset object
    lr    :    learning rate
    epoch     :    number of training epoch
    device     :   CPU or GPU
    encoder    :    GCN or GIN or GraphSAGE
    lambda_loss    :   Trade-off between degree loss and neighborhood reconstruction loss
    hidden_dim     :   latent variable dimension
    """

    in_nodes = data.edge_index[0, :]
    out_nodes = data.edge_index[1, :]

    neighbor_dict = {}
    for in_node, out_node in zip(in_nodes, out_nodes):
        if in_node.item() not in neighbor_dict:
            neighbor_dict[in_node.item()] = []
        neighbor_dict[in_node.item()].append(out_node.item())

    neighbor_num_list = []
    for i in neighbor_dict:
        neighbor_num_list.append(len(neighbor_dict[i]))

    neighbor_num_list = torch.tensor(neighbor_num_list).to(device)

    in_dim = data.x.shape[1]
    GNNModel = GNNStructEncoder(
        in_dim,
        hidden_dim,
        hidden_dim,
        2,
        sample_size,
        device=device,
        neighbor_num_list=neighbor_num_list,
        GNN_name=encoder,
        lambda_loss1=lambda_loss1,
        lambda_loss2=lambda_loss2,
        lambda_loss3=lambda_loss3,
    )
    GNNModel.to(device)
    degree_params = list(map(id, GNNModel.degree_decoder.parameters()))
    base_params = filter(lambda p: id(p) not in degree_params,
                         GNNModel.parameters())

    opt = torch.optim.Adam(
        [
            {
                "params": base_params
            },
            {
                "params": GNNModel.degree_decoder.parameters(),
                "lr": 1e-2
            },
        ],
        lr=lr,
        weight_decay=0.0003,
    )
    min_loss = float("inf")
    arg_min_loss_per_node = None

    best_auc = 0
    best_auc_contextual = 0
    best_auc_dense_structural = 0
    best_auc_joint_structural = 0
    best_auc_structure_type = 0

    loss_values = []
    for i in tqdm(range(1, epoch + 1, 1)):

        if i % loss_step == 0:
            GNNModel.lambda_loss2 = GNNModel.lambda_loss2 + 0.5
            GNNModel.lambda_loss3 = GNNModel.lambda_loss3 / 2

        loss, loss_per_node, h_loss, degree_loss, feature_loss = GNNModel(
            data.edge_index,
            data.x,
            neighbor_num_list,
            neighbor_dict,
            device=device)

        loss_per_node = loss_per_node.cpu().detach()

        h_loss = h_loss.cpu().detach()
        degree_loss = degree_loss.cpu().detach()
        feature_loss = feature_loss.cpu().detach()

        h_loss_norm = h_loss / (torch.max(h_loss) - torch.min(h_loss))
        degree_loss_norm = degree_loss / (torch.max(degree_loss) -
                                          torch.min(degree_loss))
        feature_loss_norm = feature_loss / (torch.max(feature_loss) -
                                            torch.min(feature_loss))

        comb_loss = (args.h_loss_weight * h_loss_norm +
                     args.degree_loss_weight * degree_loss_norm +
                     args.feature_loss_weight * feature_loss_norm)

        if real_loss:
            comp_loss = loss_per_node
        else:
            comp_loss = comb_loss

        from dgld.utils.evaluation import split_auc

        auc_score, a_score, s_score = split_auc(y.cpu(), comp_loss.numpy())

        # auc_score = eval_roc_auc(y.numpy(), comp_loss.numpy()) * 100
        print(
            "Dataset Name: ",
            dataset_str,
            ", AUC Score(benchmark/combined): ",
            auc_score,
        )

        # contextual_auc_score = eval_roc_auc(yc.numpy(), comp_loss.numpy()) * 100
        # print("Dataset Name: ",dataset_str, ", AUC Score (contextual): ", contextual_auc_score)

        # dense_structural_auc_score = eval_roc_auc(ys.numpy(), comp_loss.numpy()) * 100
        # print("Dataset Name: ",dataset_str, ", AUC Score (structural): ", dense_structural_auc_score)

        # joint_type_auc_score = eval_roc_auc(yj.numpy(), comp_loss.numpy()) * 100
        # print("Dataset Name: ",dataset_str, ", AUC Score (joint-type): ", joint_type_auc_score)

        # structure_type_auc_score = eval_roc_auc(ysj.numpy(), comp_loss.numpy()) * 100
        # print("Dataset Name: ",dataset_str, ", AUC Score (structure type): ", structure_type_auc_score)

        best_auc = max(best_auc, auc_score)
        # best_auc_contextual = max(best_auc_contextual, contextual_auc_score)
        # best_auc_dense_structural = max(best_auc_dense_structural, dense_structural_auc_score)
        # best_auc_joint_type = max(best_auc_joint_structural, joint_type_auc_score)
        # best_auc_structure_type = max(best_auc_structure_type, structure_type_auc_score)

        # print("===================================")
        print(dataset_str, " Best AUC Score(benchmark/combined): ", best_auc)

        # print("Dataset Name: ",dataset_str, " Best AUC Score (contextual): ", best_auc_contextual)

        # print("Dataset Name: ",dataset_str, " Best AUC Score (structural): ", best_auc_dense_structural)

        # print("Dataset Name: ",dataset_str, " Best AUC Score (joint-type): ", best_auc_joint_type)

        # print("Dataset Name: ",dataset_str, " Best AUC Score (structure type): ", best_auc_structure_type)
        # print("===================================")

        if loss < min_loss:
            min_loss = loss
            arg_min_loss_per_node = loss_per_node
        opt.zero_grad()
        loss.backward()
        opt.step()

        loss = loss.cpu().detach()
        loss_values.append(loss)

        if args.plot_loss:

            plt.plot(np.array(loss_values), "r")
            plt.show()

    return best_auc


def train_real_datasets(
    dataset_str,
    epoch_num=10,
    lr=5e-6,
    encoder="GCN",
    lambda_loss1=1e-2,
    lambda_loss2=1e-3,
    lambda_loss3=1e-3,
    sample_size=8,
    loss_step=20,
    hidden_dim=None,
    real_loss=False,
    calculate_contextual=False,
    calculate_structural=False,
):

    from dgld.utils.load_data import (
        load_data,
        load_custom_data,
        load_truth_data,
        load_local,
    )
    from dgld.utils.inject_anomalies import (
        inject_contextual_anomalies,
        inject_structural_anomalies,
    )
    from dgld.utils.common_params import Q_MAP, K, P
    import torch

    data_name = args.dataset
    truth_list = [
        "weibo",
        "tfinance",
        "tsocial",
        "reddit",
        "Amazon",
        "Class",
        "Disney",
        "elliptic",
        "Enron",
        "wiki",
        "Facebook",
        "YelpChi",
    ]
    data_path = "dgld/data"
    if data_name in truth_list:
        # graph = load_truth_data(data_path=data_path,dataset_name=data_name)
        if data_name in ["Facebook", "YelpChi"]:
            graph = load_local(data_name=data_name, raw_dir=data_path)
        else:
            graph = load_truth_data(data_path=data_path,
                                    dataset_name=data_name)
    elif data_name == "custom":
        graph = load_custom_data(data_path=data_path)
    else:
        graph = load_data(data_name)
        graph = inject_contextual_anomalies(graph=graph,
                                            k=K,
                                            p=P,
                                            q=Q_MAP[data_name],
                                            seed=4096)
        graph = inject_structural_anomalies(graph=graph,
                                            p=P,
                                            q=Q_MAP[data_name],
                                            seed=4096)
        if data_name not in ["BlogCatalog", "Flickr"]:
            from sklearn.preprocessing import LabelEncoder, MinMaxScaler

            mms = MinMaxScaler(feature_range=(0, 1))
            tmp_feat = mms.fit_transform(graph.ndata["feat"].numpy())
            graph.ndata["feat"] = torch.from_numpy(np.array(tmp_feat))

    from torch_geometric.utils import from_dgl

    data = from_dgl(graph)
    data.name = data_name
    data.num_classes = 2
    data.x = data.feat
    data.y = data.label
    data.num_nodes = graph.num_nodes()
    data.num_edges = graph.num_edges()
    data.edge_index = torch.stack(graph.edges(), dim=0)
    data = data.to(device)

    runs = 3
    import random

    seed_list = [random.randint(0, 9999999) for i in range(runs)]
    aucs = []
    # data = load_data(dataset_str)

    node_features = data.x

    if args.normalize_feat:

        node_features_min = node_features.min()
        node_features_max = node_features.max()
        node_features = (node_features - node_features_min) / node_features_max
        data.x = node_features

    # yc = []
    # ys = []
    # yj = []

    # if calculate_contextual:

    #     if dataset_str == "inj_cora":
    #         yc = data.y >> 0 & 1 # contextual outliers
    #     else:
    #         data, yc = gen_contextual_outliers(data=data,n=args.contextual_n,k=args.contextual_k)

    #     yc = yc.cpu().detach()

    # if calculate_structural:

    #     if dataset_str == "inj_cora":
    #         ys = data.y >> 1 & 1 # structural outliers
    #     else:
    #         data, ys = gen_structural_outliers(data=data,n=args.structural_n,m=args.structural_m,p=0.2)

    #     ys = ys.cpu().detach()
    #     data, yj = gen_joint_structural_outliers(data=data,n=args.structural_n,m=args.structural_m)

    # if args.use_combine_outlier:
    #     data.y = torch.logical_or(ys, yc).int()

    # ysj = torch.logical_or(ys, yj).int()
    y = data.y.bool()  # binary labels (inlier/outlier)
    y = y.cpu().detach()

    edge_index = data.edge_index.cpu()

    num_nodes = node_features.shape[0]
    self_edges = torch.tensor([[i for i in range(num_nodes)],
                               [i for i in range(num_nodes)]])
    edge_index = torch.cat([edge_index, self_edges], dim=1)
    data.edge_index = edge_index
    data = data.to(device)

    runs = 3
    import random

    seed_list = [random.randint(0, 9999999) for i in range(runs)]
    aucs = []
    for run in range(runs):
        set_seed(seed_list[run])
        auc = train(
            data,
            y,
            lr=lr,
            epoch=epoch_num,
            device=device,
            encoder=encoder,
            lambda_loss1=lambda_loss1,
            lambda_loss2=lambda_loss2,
            lambda_loss3=lambda_loss3,
            hidden_dim=hidden_dim,
            sample_size=sample_size,
            loss_step=loss_step,
            real_loss=real_loss,
            calculate_contextual=calculate_contextual,
            calculate_structural=calculate_structural,
        )
        aucs.append(auc)

    from the_utils import save_to_csv_files

    save_path = "baselines.csv"
    print(f"write result to {save_path}")
    save_to_csv_files(
        results={
            "AUC":
            f"{np.array(aucs).mean()*100:.2f}±{np.array(aucs).std()*100:.2f}",
        },
        insert_info={
            "dataset": args.dataset,
            "model": "GADNR",
        },
        append_info={"args": args.__dict__},
        csv_name=save_path,
    )


if __name__ == "__main__":
    from the_utils import tab_printer

    parser = argparse.ArgumentParser(description="parameters")
    parser.add_argument("-f")
    parser.add_argument("--dataset", type=str, default="Enron")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hid_dim", type=int, default=16)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--epoch_num", type=int, default=500)
    parser.add_argument("--lambda_loss1", type=float,
                        default=1e-2)  # neighbor reconstruction loss weight
    parser.add_argument("--lambda_loss2", type=float,
                        default=0.1)  # feature loss weight
    parser.add_argument("--lambda_loss3", type=float,
                        default=0.8)  # degree loss weight
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--encoder", type=str, default="GCN")
    parser.add_argument("--loss_step", type=int, default=100)
    parser.add_argument("--real_loss", type=bool,
                        default=False)  # use real loss or adaptive loss
    parser.add_argument("--neigh_loss", type=str, default="KL")
    parser.add_argument("--h_loss_weight", type=float,
                        default=1.0)  # adaptive loss weight for h_loss
    parser.add_argument("--feature_loss_weight", type=float,
                        default=2.0)  # adaptive loss weight for feature loss
    parser.add_argument("--degree_loss_weight", type=float,
                        default=1.0)  # adaptive loss weight for degree loss
    parser.add_argument("--calculate_contextual", type=bool, default=False)
    parser.add_argument("--contextual_n", type=int, default=3)
    parser.add_argument("--contextual_k", type=int, default=25)
    parser.add_argument("--calculate_structural", type=bool, default=False)
    parser.add_argument("--structural_n", type=int, default=3)
    parser.add_argument("--structural_m", type=int, default=25)
    parser.add_argument("--use_combine_outlier", type=bool, default=False)
    parser.add_argument("--plot_loss", type=bool, default=False)
    parser.add_argument("--normalize_feat", type=bool, default=False)

    args = parser.parse_args()

    tab_printer(args.__dict__)

    from the_utils import set_device

    device = set_device(args.gpu)

    print("GAD-NR: Graph Anomaly Detection via Neighborhood Reconstruction")
    print(
        "Dataset: ",
        args.dataset,
        "lr:",
        args.lr,
        "lambda_loss1 (neighbor):",
        args.lambda_loss1,
        "lambda_loss2 (feature):",
        args.lambda_loss2,
        "lambda_loss3 (degree):",
        args.lambda_loss3,
        "sample_size:",
        args.sample_size,
        "dimension:",
        args.dimension,
        "encoder:",
        args.encoder,
        "loss_step:",
        args.loss_step,
        "real_loss:",
        args.real_loss,
        "h_loss_weight:",
        args.h_loss_weight,
        "feature_loss_weight",
        args.feature_loss_weight,
        "degree_loss_weight:",
        args.degree_loss_weight,
        "calculate_contextual",
        args.calculate_contextual,
        "calculate_structural",
        args.calculate_structural,
    )

    dataset_str = args.dataset
    train_real_datasets(
        dataset_str=dataset_str,
        lr=args.lr,
        epoch_num=args.epoch_num,
        lambda_loss1=args.lambda_loss1,
        lambda_loss2=args.lambda_loss2,
        lambda_loss3=args.lambda_loss3,
        encoder=args.encoder,
        sample_size=args.sample_size,
        loss_step=args.loss_step,
        hidden_dim=args.hid_dim,
        real_loss=args.real_loss,
        calculate_contextual=args.calculate_contextual,
        calculate_structural=args.calculate_structural,
    )
