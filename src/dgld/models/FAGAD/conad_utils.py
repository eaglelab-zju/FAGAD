import math
import os
import sys
from secrets import choice

import dgl
import numpy as np
import torch
from scipy.linalg import hadamard
from torch import nn

current_file_name = __file__
current_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(current_file_name)))
)
if current_dir not in sys.path:
    sys.path.append(current_dir)


def set_subargs(parser):
    parser.add_argument("--num_epoch", type=int, default=100, help="Training epoch")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.9, help="balance parameter")
    parser.add_argument(
        "--eta", type=float, default=0.7, help="Attribute penalty balance parameter"
    )
    parser.add_argument(
        "--contrast_type",
        type=str,
        default="siamese",
        choices=["siamese", "triplet"],
        help="categories of contrastive loss function",
    )
    parser.add_argument("--rate", type=float, default=0.2, help="rate of anomalies")
    parser.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="parameter of the contrastive loss function",
    )
    parser.add_argument(
        "--batch_size", type=int, default=0, help="size of training batch"
    )
    parser.add_argument(
        "--num_added_edge",
        type=int,
        default=50,
        help="parameter for generating high-degree anomalies",
    )
    parser.add_argument(
        "--surround",
        type=int,
        default=50,
        help="parameter for generating outlying anomalies",
    )
    parser.add_argument(
        "--scale_factor",
        type=float,
        default=10,
        help="parameter for generating disproportionate anomalies",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=64, help="dimension of embedding"
    )
    parser.add_argument(
        "--struct_dec_act",
        type=str,
        default=None,
        help="structure decoder non-linear activation function",
    )


def get_subargs(args):
    final_args_dict = {
        "dataset": args.dataset,
        "seed": args.seed,
        "model": {
            "feat_size": args.feat_dim,
            "embedding_dim": args.embedding_dim,
            "struct_dec_act": args.struct_dec_act,
        },
        "fit": {
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "num_epoch": args.num_epoch,
            "device": args.device,
            "eta": args.eta,
            "alpha": args.alpha,
            "rate": args.rate,
            "contrast_type": args.contrast_type,
            "margin": args.margin,
            "batch_size": args.batch_size,
        },
        "predict": {
            "device": args.device,
            "alpha": args.alpha,
            "batch_size": args.batch_size,
        },
    }
    return final_args_dict, args


def loss_func(a, a_hat, x, x_hat, alpha):
    # # adjacency matrix reconstruction loss
    # struct_norm = torch.linalg.norm(a-a_hat, dim=1)
    # struct_loss = torch.mean(struct_norm)
    # # feature matrix reconstruction loss
    # feat_norm = torch.linalg.norm(x-x_hat, dim=1)
    # feat_loss = torch.mean(feat_norm)
    # # total reconstruction loss
    # loss = (1-alpha) * struct_norm + alpha * feat_norm
    # return loss, struct_loss, feat_loss
    # Attribute reconstruction loss
    diff_attribute = torch.pow(x_hat - x, 2)
    attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
    attribute_cost = torch.mean(attribute_reconstruction_errors)

    # structure reconstruction loss
    diff_structure = torch.pow(a_hat - a, 2)
    structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
    structure_cost = torch.mean(structure_reconstruction_errors)

    cost = (
        alpha * attribute_reconstruction_errors
        + (1 - alpha) * structure_reconstruction_errors
    )
    return cost, structure_cost, attribute_cost


# def train_step(model, optimizer, g_orig, g_aug, alpha, eta):
#     model.train()
#     model.to('cpu')
#     g_orig=g_orig.to('cpu')
#     g_aug=g_aug.to('cpu')
#     feat_orig, feat_aug = g_orig.ndata['feat'], g_aug.ndata['feat']
#     label = g_aug.ndata['label']
#     #byol contrastive loss
#     contrast_loss = model.byol_forward(g_orig, feat_orig, g_aug, feat_aug)
#     # recontruct loss
#     adj_orig = g_orig.adj().to_dense()#.to(label.device)
#     h = model.online_network(g_orig, feat_orig)
#     a_hat, x_hat = model.reconstruct(g_orig, h)
#     recon_loss, struct_loss, feat_loss = loss_func(adj_orig, a_hat, feat_orig, x_hat, alpha)
#     # total loss
#     loss = eta * contrast_loss + (1-eta) * recon_loss.mean()
#     # backward
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     model._update_target_network_parameters()
#     return loss,contrast_loss,struct_loss, feat_loss


def train_step(model, optimizer, g_orig, g_aug, alpha, eta):
    model.train()
    feat_orig, feat_aug = g_orig.ndata["feat"], g_aug.ndata["feat"]
    label = g_aug.ndata["label"]
    # byol contrastive loss
    contrast_loss = model.byol_forward(g_orig, feat_orig, g_aug, feat_aug)
    # recontruct loss
    adj_orig = g_orig.adj().to_dense().to(label.device)
    h = model.online_network(g_orig, feat_orig)
    a_hat, x_hat = model.reconstruct(g_orig, h)
    recon_loss, struct_loss, feat_loss = loss_func(
        adj_orig, a_hat, feat_orig, x_hat, alpha
    )
    # total loss
    loss = eta * contrast_loss + (1 - eta) * recon_loss.mean()
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model._update_target_network_parameters()
    return loss, contrast_loss, struct_loss, feat_loss


# def test_step(model, graph, alpha):
#     model.eval()
#     model.to('cpu')
#     graph = graph.to('cpu')
#     feat = graph.ndata['feat']
#     h = model.online_network(graph, feat)
#     a_hat, x_hat = model.reconstruct(graph, h)
#     a_hat, x_hat = a_hat, x_hat
#     # print(feat.device)
#     adj_orig = graph.adj_external().to_dense().to(feat.device)
#     feat_orig = feat.detach()
#     recon_loss, struct_loss, feat_loss = loss_func(adj_orig, a_hat, feat_orig, x_hat, alpha)
#     score = recon_loss.cpu().detach().numpy()
#     return score


def test_step(model, graph, alpha):
    model.eval()
    feat = graph.ndata["feat"]
    h = model.online_network(graph, feat)
    a_hat, x_hat = model.reconstruct(graph, h)
    a_hat, x_hat = a_hat, x_hat
    # print(feat.device)
    adj_orig = graph.adj_external().to_dense().to(feat.device)
    feat_orig = feat.detach()
    recon_loss, struct_loss, feat_loss = loss_func(
        adj_orig, a_hat, feat_orig, x_hat, alpha
    )
    score = recon_loss.cpu().detach().numpy()
    return score


def train_step_batch(
    model: nn.Module, optimizer, g_orig, g_aug, alpha, eta, batch_size, device
):
    model.train()
    node_list = g_orig.nodes()
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=3)
    dataloader_orig = dgl.dataloading.DataLoader(
        g_orig,
        node_list,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    dataloader_aug = dgl.dataloading.DataLoader(
        g_aug,
        node_list,
        sampler,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    epoch_loss = 0
    for (input_nodes_orig, output_nodes_orig, blocks_orig), (
        input_nodes_aug,
        output_nodes_aug,
        blocks_aug,
    ) in zip(dataloader_orig, dataloader_aug):
        optimizer.zero_grad()
        label = blocks_aug[-1].dstdata["label"]
        output_nodes_num = blocks_orig[-1].number_of_dst_nodes()
        adj_orig = (
            dgl.block_to_graph(blocks_orig[-1])
            .adj()
            .to_dense()[:output_nodes_num, :output_nodes_num]
        )

        blocks_orig = [b.to(device) for b in blocks_orig]
        blocks_aug = [b.to(device) for b in blocks_aug]
        label = label.to(device)
        adj_orig = adj_orig.to(device)

        input_feat_orig = blocks_orig[0].srcdata["feat"]
        input_feat_aug = blocks_aug[0].srcdata["feat"]

        # contrastive loss
        h_orig = model.online_network(blocks_orig, input_feat_orig)
        h_aug = model.online_network(blocks_aug, input_feat_aug)

        # print(h_orig.shape, h_aug.shape, adj_orig.shape, label.shape)
        contrast_loss = model.byol_forward(
            h_orig[:output_nodes_num], h_aug[:output_nodes_num], label, adj_orig
        )
        # recontruct loss
        a_hat, x_hat = model(blocks_orig, input_feat_orig)
        output_feat_orig = blocks_orig[-1].dstdata["feat"]
        recon_loss, struct_loss, feat_loss = loss_func(
            adj_orig,
            a_hat[:output_nodes_num, :output_nodes_num],
            output_feat_orig,
            x_hat[:output_nodes_num],
            alpha,
        )
        # total loss
        loss = eta * contrast_loss + (1 - eta) * recon_loss.mean()
        # backward
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_size
        print(loss.item())

    return epoch_loss


def test_step_batch(model, graph, alpha, batch_size, device):
    model.eval()
    model.train(False)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(num_layers=3)
    dataloader = dgl.dataloading.DataLoader(
        graph,
        graph.nodes(),
        sampler,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    score = np.zeros(graph.num_nodes())

    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(device) for b in blocks]
        feat = blocks[0].srcdata["feat"]
        a_hat, x_hat = model(blocks, feat)

        adj_orig = dgl.node_subgraph(graph, output_nodes).adj().to_dense().to(device)
        feat_orig = blocks[-1].dstdata["feat"]
        output_nodes_num = blocks[-1].number_of_dst_nodes()
        recon_loss, struct_loss, feat_loss = loss_func(
            adj_orig,
            a_hat[:output_nodes_num, :output_nodes_num],
            feat_orig,
            x_hat[:output_nodes_num],
            alpha,
        )
        score[output_nodes] = recon_loss.cpu().detach().numpy()

    return score


def Identity_Init_on_matrix(matrix_tensor):
    # Definition 1 in the paper
    # See https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.eye_ for details. Preserves the identity of the inputs in Linear layers, where as many inputs are preserved as possible, the same as partial identity matrix.

    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)

    init_matrix = torch.nn.init.eye_(torch.empty(m, n))

    return init_matrix


def ZerO_Init_on_matrix(matrix_tensor):
    # Algorithm 1 in the paper. https://arxiv.org/abs/2110.12661

    m = matrix_tensor.size(0)
    n = matrix_tensor.size(1)

    if m <= n:
        init_matrix = torch.nn.init.eye_(torch.empty(m, n))
    elif m > n:
        clog_m = math.ceil(math.log2(m))
        p = 2 ** (clog_m)
        init_matrix = (
            torch.nn.init.eye_(torch.empty(m, p))
            @ (torch.tensor(hadamard(p)).float() / (2 ** (clog_m / 2)))
            @ torch.nn.init.eye_(torch.empty(p, n))
        )

    return init_matrix
