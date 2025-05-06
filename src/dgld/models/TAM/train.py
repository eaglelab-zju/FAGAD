# -*- coding: utf-8 -*-
import argparse
import os
import random
import time

import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from the_utils import make_parent_dirs
from the_utils import set_device
from the_utils import set_seed
from the_utils import tab_printer
from tqdm import tqdm

from .model import Model
from .utils import *

# Set argument
parser = argparse.ArgumentParser(
    description="Truncated Affinity Maximization for Graph Anomaly Detection"
)
parser.add_argument("--dataset", type=str, default="Amazon")
parser.add_argument("--gpu", type=str, default="6", help="GPU id")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--num_epoch", type=int)
parser.add_argument("--drop_prob", type=float, default=0.0)
parser.add_argument("--subgraph_size", type=int, default=15)
parser.add_argument("--readout", type=str, default="avg")  # max min avg  weighted_sum
parser.add_argument("--margin", type=int, default=2)
parser.add_argument("--negsamp_ratio", type=int, default=2)
parser.add_argument("--cutting", type=int, default=3)  # 3 5 8 10
parser.add_argument("--N_tree", type=int, default=3)  # 3 5 8 10
parser.add_argument("--lamda", type=float, default=0)  # 0  0.5  1
args = parser.parse_args()

# args.lr = 1e-5
args.num_epoch = 500

# os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, [args.gpu]))
# os.environ["KMP_DUPLICATE_LnIB_OK"] = "TRUE"
device = set_device(args.gpu)
device_ids = [args.gpu]

# print("Dataset: ", args.dataset)

# Set random seed
# dgl.random.seed(args.seed)
# np.random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# random.seed(args.seed)
# os.environ["PYTHONHASHSEED"] = str(args.seed)
# os.environ["OMP_NUM_THREADS"] = "1"
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# Load and preprocess data
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
from dgld.utils.evaluation import split_auc
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
        graph = load_truth_data(data_path=data_path, dataset_name=data_name)
elif data_name == "custom":
    graph = load_custom_data(data_path=data_path)
else:
    graph = load_data(data_name)
    graph = inject_contextual_anomalies(
        graph=graph, k=K, p=P, q=Q_MAP[data_name], seed=4096
    )
    graph = inject_structural_anomalies(graph=graph, p=P, q=Q_MAP[data_name], seed=4096)
    print(
        min(graph.ndata["label"]), max(graph.ndata["label"]), graph.ndata["label"].shape
    )
    if data_name not in ["BlogCatalog", "Flickr"]:
        from sklearn.preprocessing import LabelEncoder, MinMaxScaler

        mms = MinMaxScaler(feature_range=(0, 1))
        tmp_feat = mms.fit_transform(graph.ndata["feat"].numpy())
        graph.ndata["feat"] = torch.from_numpy(np.array(tmp_feat))

# pca = PCA(n_components=10)
# label = graph.ndata['label']
# features = graph.ndata['feat']

# adj, features, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)
adj = graph.adj_external(scipy_fmt="csr")
features = graph.ndata["feat"]
raw_features = features
ano_label = graph.ndata["label"]

# if args.dataset in ["Amazon", "YelpChi", "Amazon-all", "YelpChi-all"]:
#     features, _ = preprocess_features(features)
#     raw_features = features
# else:
#     raw_features = features.todense()
#     features = raw_features

dgl_graph = graph
nb_nodes = features.shape[0]
ft_size = features.shape[1]
raw_adj = adj
raw_adj = (raw_adj + sp.eye(adj.shape[0])).todense()
adj = (adj + sp.eye(adj.shape[0])).todense()
raw_features = torch.FloatTensor(raw_features[np.newaxis])
features = torch.FloatTensor(features[np.newaxis])
adj = torch.FloatTensor(adj[np.newaxis])
raw_adj = torch.FloatTensor(raw_adj[np.newaxis])


def reg_edge(emb, adj):
    emb = emb / torch.norm(emb, dim=-1, keepdim=True)
    sim_u_u = torch.mm(emb, emb.T)
    adj_inverse = 1 - adj
    sim_u_u = sim_u_u * adj_inverse
    sim_u_u_no_diag = torch.sum(sim_u_u, 1)
    row_sum = torch.sum(adj_inverse, 1)
    r_inv = torch.pow(row_sum, -1)
    r_inv[torch.isinf(r_inv)] = 0.0
    sim_u_u_no_diag = sim_u_u_no_diag * r_inv
    loss_reg = torch.sum(sim_u_u_no_diag)

    return loss_reg


def max_message(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)

    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix
    sim_matrix[torch.isinf(sim_matrix)] = 0
    sim_matrix[torch.isnan(sim_matrix)] = 0

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0

    message = torch.sum(sim_matrix, 1)

    message = message * r_inv
    # message = (message - torch.min(message)) / (torch.max(message) - torch.min(message))
    # message[torch.isinf(message)] = 0.
    # message[torch.isnan(message)] = 0.
    return -torch.sum(message), message


def inference(feature, adj_matrix):
    feature = feature / torch.norm(feature, dim=-1, keepdim=True)
    sim_matrix = torch.mm(feature, feature.T)
    sim_matrix = torch.squeeze(sim_matrix) * adj_matrix

    row_sum = torch.sum(adj_matrix, 0)
    r_inv = torch.pow(row_sum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.0
    message = torch.sum(sim_matrix, 1)
    message = message * r_inv
    return message


start = time.time()
# Train model
# with tqdm(total=args.num_epoch) as pbar:
#     pbar.set_description("Training")

score_list = []
new_adj_list = []
for n_t in range(args.N_tree):
    new_adj_list.append(raw_adj)
all_cut_adj = torch.cat(new_adj_list)
origin_degree = torch.sum(torch.squeeze(raw_adj), 0)
from pathlib import Path

dis_path = Path(f"./tmp/{data_name}")
if dis_path.exists():
    print(f"<<<<<<Load dis_array from {dis_path}<<<<<")
    dis_array = torch.load(dis_path, device)
else:
    print("<<<<<<Start to calculate distance<<<<<")
    dis_array = calc_distance(raw_adj[0, :, :], raw_features[0, :, :])
    make_parent_dirs(dis_path)
    torch.save(dis_array, dis_path, pickle_protocol=4)
    print("<<<<<<End<<<<<")

runs = 3
seed_list = [random.randint(0, 99999) for i in range(runs)]

aucs = []
tab_printer(args.__dict__)
et = []
for run in range(runs):
    index = 0
    message_mean_list = []
    seed = seed_list[run]
    set_seed(seed)
    # Initialize model and optimiser
    optimiser_list = []
    model_list = []
    for i in range(args.cutting * args.N_tree):
        model = Model(
            ft_size, args.embedding_dim, "prelu", args.negsamp_ratio, args.readout
        )
        optimiser = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        if torch.cuda.is_available():
            model = model.cuda()
            optimiser_list.append(optimiser)
            model_list.append(model)

    criterion = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        print("Using CUDA")
        features = features.cuda()
        raw_features = raw_features.cuda()
        adj = adj.cuda()
        raw_adj = raw_adj.cuda()
        all_cut_adj = all_cut_adj.cuda()

    time_st = time.time()
    for n_cut in range(args.cutting):
        print("n_cut.{}".format(n_cut))
        feat_list = []
        message_list = []
        for n_t in range(args.N_tree):
            cut_adj = graph_nsgt(dis_array, all_cut_adj[n_t, :, :])
            cut_adj = cut_adj.unsqueeze(0)
            optimiser_list[index].zero_grad()
            model_list[index].train()
            print("<<<< cutting num .{}<<<<<<".format(n_cut))
            adj_norm = normalize_adj_tensor(cut_adj)

            for epoch in range(args.num_epoch):
                all_idx = list(range(nb_nodes))

                node_emb, feat1, feat2 = model_list[index].forward(features, adj_norm)
                # maximize the message flow
                loss, message_sum1 = max_message(node_emb[0, :, :], raw_adj[0, :, :])

                message_sum = inference(node_emb[0, :, :], raw_adj[0, :, :])
                reg_loss = reg_edge(feat1[0, :, :], raw_adj[0, :, :])
                loss = loss + args.lamda * reg_loss
                loss.backward()
                optimiser_list[index].step()

                loss = loss.detach().cpu().numpy()

                if epoch % 50 == 0:
                    print("mean_loss is {}".format(loss))
            message_list.append(torch.unsqueeze(message_sum, 0))
            all_cut_adj[n_t, :, :] = torch.squeeze(cut_adj)
            index += 1

        for mes in message_list:
            mes = np.array(torch.squeeze(mes).cpu().detach())
            mes = 1 - normalize_score(mes)
            # auc = roc_auc_score(ano_label, mes)
            auc, a_score, s_score = split_auc(ano_label, mes)
            print("{} AUC:{:.4f}".format(args.dataset, auc))

        message_list = torch.mean(torch.cat(message_list), 0)
        message_mean_list.append(torch.unsqueeze(message_list, 0))

        message = np.array(message_list.cpu().detach())
        adj_array = np.array(raw_adj[0, :, :].cpu().detach())

        message = 1 - normalize_score(message)

        # draw_pdf(1 - message, ano_label, args.dataset)
        score = message

        auc, a_score, s_score = split_auc(ano_label, score)
        # auc = roc_auc_score(ano_label, score)

        # AP = average_precision_score(ano_label,
        #                              score,
        #                              average="macro",
        #                              pos_label=1,
        #                              sample_weight=None)
        # print("AP:", AP)
        print("{} AUC:{:.4f}".format(args.dataset, auc))
        message_mean_cut = torch.mean(torch.cat(message_mean_list), 0)
        message_mean = np.array(message_mean_cut.cpu().detach())
        message_mean = 1 - normalize_score(message_mean)
        score = message_mean
        # auc = roc_auc_score(ano_label, score)
        auc, a_score, s_score = split_auc(ano_label, score)
        # AP = average_precision_score(ano_label,
        #                              score,
        #                              average="macro",
        #                              pos_label=1,
        #                              sample_weight=None)
        # print("AP:", AP)
        print("{} AUC:{:.4f}".format(args.dataset, auc))

    et.append((time.time() - time_st) / (epoch + 1) * 10)
    aucs.append(auc)

    end = time.time()
    print(end - start)


import numpy as np
from the_utils import save_to_csv_files

elapsed_time = np.array(et)
save_to_csv_files(
    {
        "time": f"{elapsed_time.mean():.2f}±{elapsed_time.std():.2f}",
    },
    "time.csv",
    insert_info={
        "model": "TAM",
        "dataet": data_name,
    },
)

from the_utils import save_to_csv_files

save_path = "baselines.csv"
print(f"write result to {save_path}")
save_to_csv_files(
    results={
        "AUC": f"{np.array(aucs).mean()*100:.2f}±{np.array(aucs).std()*100:.2f}",
    },
    insert_info={
        "dataset": args.dataset,
        "model": "TAM",
    },
    append_info={
        "args": args.__dict__,
    },
    csv_name=save_path,
)
