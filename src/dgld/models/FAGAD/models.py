import math
import time
import warnings
from copy import deepcopy

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import ChebConv
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import GraphConv
from scipy.linalg import hadamard

from .BernNet import BernNet
from .conad_utils import *
from .FAGCN import FAGCN

warnings.filterwarnings("ignore")

import sys

sys.path.append("../../")
from dgld.utils.early_stopping import EarlyStopping


class CONAD(nn.Module):

    def __init__(
        self,
        in_feats,
        hid_feats=128,
        mlp_hidden_size=128,
        projection_size=128,
        struct_dec_act=None,
        m=0.99,
        k=2,
        encoder_name="FAGCN",
        init="random",
        dropout=0.5,
    ):
        super(CONAD, self).__init__()
        self.model = CONAD_Base(
            in_feats,
            hid_feats,
            mlp_hidden_size,
            projection_size,
            struct_dec_act,
            m,
            k,
            encoder_name,
            init,
            dropout=dropout,
        )

    def fit(
        self,
        graph,
        lr=1e-3,
        weight_decay=0.0,
        alpha=0.9,
        eta=0.7,
        num_epoch=100,
        device="cpu",
        batch_size=0,
    ):
        print("*" * 20, "training", "*" * 20)

        if torch.cuda.is_available() and device != "cpu":
            device = torch.device("cuda:" + device)
            print("Using gpu!!!")
        else:
            device = torch.device("cpu")
            print("Using cpu!!!")

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        graph = graph.remove_self_loop()

        g_orig = graph.add_self_loop()
        g_aug = g_orig

        self.model.to(device)

        early_stop = EarlyStopping(early_stopping_rounds=100, patience=20)

        if batch_size == 0:
            print("full graph training!!!")
            g_orig = g_orig.to(device)
            g_aug = g_aug.to(device)
            time_st = time.time()
            for epoch in range(num_epoch):

                loss_epoch, contrast_loss, struct_loss, feat_loss = train_step(
                    self.model, optimizer, g_orig, g_aug, alpha=alpha, eta=eta
                )
                print(
                    f"Epoch:{epoch}",
                    f",loss={loss_epoch.item()}",
                    f",contrast_loss={contrast_loss.item()}",
                    f",struct_loss={struct_loss.item()}",
                    f",feat_loss={feat_loss.item()}",
                )

        else:
            print("batch graph training!!!")
            time_st = time.time()
            for epoch in range(num_epoch):
                loss = train_step_batch(
                    self.model, optimizer, g_orig, g_aug, alpha, eta, batch_size, device
                )
                print("Epoch:", "%04d" % (epoch), "train/loss=", "{:.5f}".format(loss))

                early_stop(loss, self.model)
                if early_stop.isEarlyStopping():
                    print(f"Early stopping in round {epoch}")
                    break
        return (time.time() - time_st) / (epoch + 1) * 10

    def predict(
        self,
        graph,
        alpha=0.9,
        device="cpu",
        batch_size=0,
    ):
        print("*" * 20, "predict", "*" * 20)

        if torch.cuda.is_available() and device != "cpu":
            device = torch.device("cuda:" + device)
            print("Using gpu!!!")
        else:
            device = torch.device("cpu")
            print("Using cpu!!!")

        if batch_size == 0:
            graph = graph.remove_self_loop().add_self_loop().to(device)
            self.model.to(device)
            predict_score = test_step(self.model, graph, alpha=alpha)
        else:
            graph = graph.remove_self_loop().add_self_loop()
            self.model.to(device)
            predict_score = test_step_batch(
                self.model, graph, alpha, batch_size, device
            )

        return predict_score


class CONAD_Base(nn.Module):

    def __init__(
        self,
        in_feats,
        hid_feats=128,
        mlp_hidden_size=64,
        projection_size=64,
        struct_dec_act=None,
        m=0.99,
        k=2,
        encoder_name="FAGCN",
        init="random",
        dropout=0.5,
    ):
        super(CONAD_Base, self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.m = m
        self.encoder_name = encoder_name
        act = eval(f"F.{struct_dec_act}")
        self.online_network = Encoder(
            in_feats,
            hid_feats,
            mlp_hidden_size,
            projection_size,
            k,
            encoder_name,
            activation=act,
            dropout=dropout,
        )
        self.target_network = Encoder(
            in_feats,
            hid_feats,
            mlp_hidden_size,
            projection_size,
            k,
            encoder_name,
            activation=act,
            dropout=dropout,
        )
        self.initializes_target_network()
        # predictor network
        self.predictor = MLPHead(projection_size, mlp_hidden_size, projection_size)

        # NOTE: FAGCN is the Frequency Self-Adaptation Graph Neural Network
        if encoder_name == "FAGCN":
            self.attr_decoder = FAGCN(projection_size, in_feats, k)
        elif encoder_name == "ChebNet":
            self.attr_decoder = ChebConv(
                projection_size, in_feats, k + 1, activation=act
            )
        elif encoder_name == "GCN":
            self.attr_decoder = GraphConv(projection_size, in_feats, activation=act)
        elif encoder_name == "GAT":
            self.attr_decoder = GATConv(
                projection_size, in_feats, num_heads=2, activation=act
            )
        elif encoder_name == "BernNet":
            self.attr_decoder = BernNet(projection_size, in_feats, d=k + 1)

        self.struct_decoder = lambda h: h @ h.T
        self.struct_dec_act = struct_dec_act

        for module in self.modules():
            init_weights(module, init=init)

        # print(self.online_network.encoder.conv.linear.weight.data)
        # print(self.online_network.encoder.linear.weight.data)
        # print(self.target_network.encoder.conv.linear.weight.data)

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def initializes_target_network(self):
        # init momentum network as encoder net
        for param_q, param_k in zip(
            self.online_network.parameters(), self.target_network.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def embed_batch(self, blocks, h):
        """
        Compute embeddings for mini-batch graph training

        Parameters
        ----------
        g : list
            graph dataset
        h : torch.Tensor
            features of nodes

        Returns
        -------
        torch.Tensor
            embeddings of nodes
        """
        for i, layer in enumerate(self.online_network):
            h = layer(blocks[i], h)
            h = h.flatten(1) if i == 0 else h.mean(1)
        return h

    def embed(self, g, h):
        if isinstance(g, list):
            return self.embed_batch(g, h)
        for i, layer in enumerate(self.shared_encoder):
            h = layer(g, h)
        print("h", h)
        return h

    def reconstruct(self, g, h):
        # reconstruct attribute matrix
        if self.encoder_name == "GAT":
            x_hat = self.attr_decoder(g, h).mean(1)
        else:
            x_hat = self.attr_decoder(g, h)
        # reconstruct adjacency matrix
        a_hat = self.struct_decoder(h)
        if self.struct_dec_act is not None:
            a_hat = eval(f"torch.{self.struct_dec_act}(a_hat)")
        return a_hat, x_hat

    def byol_forward(self, g1, h1, g2, h2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(g1, h1))

        # compute key features
        with torch.no_grad():
            targets_to_view_1 = self.target_network(g2, h2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        return loss.mean()


class Encoder(nn.Module):

    def __init__(
        self,
        in_feats,
        hid_feats=128,
        mlp_hidden_size=64,
        projection_size=64,
        k=2,
        encoder_name="FAGCN",
        activation=None,
        dropout=0.5,
    ):
        super().__init__()
        self.encoder_name = encoder_name
        if encoder_name == "FAGCN":
            self.encoder = FAGCN(in_feats, hid_feats, k, dropout=dropout)
        elif encoder_name == "GAT":
            self.encoder = GATConv(
                in_feats, hid_feats, num_heads=2, activation=activation
            )
        elif encoder_name == "ChebNet":
            self.encoder = ChebConv(
                in_feats, hid_feats, k + 1, activation=activation
            )  # +1 because low pass
        elif encoder_name == "GCN":
            self.encoder = GraphConv(in_feats, hid_feats, activation=activation)
        elif encoder_name == "BernNet":
            self.encoder = BernNet(in_feats, hid_feats, k + 1)

        self.projection = MLPHead(hid_feats, mlp_hidden_size, projection_size)

    def forward(self, g, x):
        if self.encoder_name == "GAT":
            x = self.encoder(g, x).mean(1)
        else:
            x = self.encoder(g, x)
        x = self.projection(x)
        return x


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


def init_weights(module: nn.Module, init="random") -> None:
    if isinstance(module, nn.Linear):
        print("model init using::::::", init)
        if init == "random":
            nn.init.xavier_uniform_(module.weight.data)
        elif init == "zero":
            module.weight.data = ZerO_Init_on_matrix(module.weight.data)
        elif init == "identity":
            module.weight.data = Identity_Init_on_matrix(module.weight.data)

        if module.bias is not None:
            module.bias.data.fill_(0.0)


# for test
if __name__ == "__main__":
    from dgld.utils.evaluation import split_auc
    from dgld.utils.common import seed_everything, csv2file
    from dgld.utils.argparser import parse_all_args
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
    import random
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from the_utils import save_to_csv_files, set_device, set_seed

    import argparse

    parser = argparse.ArgumentParser(
        prog="All", description="Parameters for Baseline Method"
    )
    # experiment parameter
    parser.add_argument("--gpu", type=str, default="6", help="GPU id")
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="The number of runs of task with same parmeter",
    )
    parser.add_argument(
        "--dataset", type=str, default="cora", help="Dataset used in the experiment"
    )
    parser.add_argument("--model_init", type=str, default="zero", help="model_init")
    parser.add_argument("--hid_feats", type=int, default=512, help="hid_feats")
    parser.add_argument("--mlp_hidden_dic", type=int, default=1024, help="hid_feats")
    parser.add_argument("--projection_dic", type=int, default=1024, help="hid_feats")
    parser.add_argument(
        "--struct_dec_act", type=str, default="relu", help="struct_dec_act"
    )
    parser.add_argument("--k_dic", type=int, default=1, help="k_dic")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="learning rate")
    parser.add_argument("--alpha", type=float, default=0.2, help="alpha")
    parser.add_argument("--eta", type=float, default=0.1, help="eta")
    parser.add_argument("--dropout_dic", type=float, default=0.0, help="eta")
    parser.add_argument("--num_epoch", type=int, default=10, help="num_epoch")

    args = parser.parse_args()
    set_device(args.gpu)

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
    # save_path = 'dgld/models/SaGA_v1/results/UFA_elliptic_0301.csv'
    save_path = "dgld/models/SaGA_v1/results/SAG.csv"
    runs = args.runs
    seed_list = [random.randint(0, 99999) for i in range(runs)]

    # Encoder_name_list = ['ChebNet','GAT','GCN','BernNet']
    # Encoder_name_list = ['ChebNet']
    Encoder_name_list = ["FAGCN"]
    # model_init = 'identity'#zero,identity,random
    # Data_name_list=['Flickr','BlogCatalog','wiki','reddit','Amazon','Enron']# BlogCatalog/Flickr
    # Data_name_list=['Enron','Flickr','wiki','reddit','BlogCatalog','Amazon']
    # Data_name_list=["Facebook", "YelpChi"]
    Data_name_list = [args.dataset]
    # Data_name_list=['elliptic']
    for data_name in Data_name_list:
        # best params
        model_init_dic = {
            "Amazon": args.model_init,
            "Facebook": args.model_init,
            "YelpChi": args.model_init,
            "reddit": args.model_init,
            "Enron": args.model_init,
            "wiki": args.model_init,
            "BlogCatalog": args.model_init,
            "Flickr": args.model_init,
            "tfinance": args.model_init,
        }

        hid_feats_dic = {
            "Amazon": args.hid_feats,
            "Facebook": args.hid_feats,
            "YelpChi": args.hid_feats,
            "reddit": args.hid_feats,
            "Enron": args.hid_feats,
            "wiki": args.hid_feats,
            "BlogCatalog": args.hid_feats,
            "Flickr": args.hid_feats,
            "tfinance": args.hid_feats,
        }
        mlp_hidden_dic = {
            "Amazon": args.mlp_hidden_dic,
            "Facebook": args.mlp_hidden_dic,
            "YelpChi": args.mlp_hidden_dic,
            "reddit": args.mlp_hidden_dic,
            "Enron": args.mlp_hidden_dic,
            "wiki": args.mlp_hidden_dic,
            "BlogCatalog": args.mlp_hidden_dic,
            "Flickr": args.mlp_hidden_dic,
            "tfinance": args.mlp_hidden_dic,
        }
        projection_dic = {
            "Amazon": args.projection_dic,
            "Facebook": args.projection_dic,
            "YelpChi": args.projection_dic,
            "reddit": args.projection_dic,
            "Enron": args.projection_dic,
            "wiki": args.projection_dic,
            "BlogCatalog": args.projection_dic,
            "Flickr": args.projection_dic,
            "tfinance": args.projection_dic,
        }
        struct_dec_act_dic = {
            "Amazon": args.struct_dec_act,
            "Facebook": args.struct_dec_act,
            "YelpChi": args.struct_dec_act,
            "reddit": args.struct_dec_act,
            "Enron": args.struct_dec_act,
            "wiki": args.struct_dec_act,
            "BlogCatalog": args.struct_dec_act,
            "Flickr": args.struct_dec_act,
            "tfinance": args.struct_dec_act,
        }
        m_dic = {
            "Amazon": 0.9,
            "Facebook": 0.9,
            "YelpChi": 0.9,
            "reddit": 0.99,
            "Enron": 0.9999,
            "wiki": 0.9,
            "BlogCatalog": 0.9999,
            "Flickr": 0.9,
            "tfinance": 0.9,
        }
        k_dic = {
            "Amazon": args.k_dic,
            "Facebook": args.k_dic,
            "YelpChi": args.k_dic,
            "reddit": args.k_dic,
            "Enron": args.k_dic,
            "wiki": args.k_dic,
            "BlogCatalog": args.k_dic,
            "Flickr": args.k_dic,
            "tfinance": args.k_dic,
        }
        lr_dic = {
            "Amazon": args.lr,
            "Facebook": args.lr,
            "YelpChi": args.lr,
            "reddit": args.lr,
            "Enron": args.lr,
            "wiki": args.lr,
            "BlogCatalog": args.lr,
            "Flickr": args.lr,
            "tfinance": args.lr,
        }
        alpha_dic = {
            "Amazon": args.alpha,
            "Facebook": args.alpha,
            "YelpChi": args.alpha,
            "reddit": args.alpha,
            "Enron": args.alpha,
            "wiki": args.alpha,
            "BlogCatalog": args.alpha,
            "Flickr": args.alpha,
            "tfinance": args.alpha,
        }
        eta_dic = {
            "Amazon": args.eta,
            "Facebook": args.eta,
            "YelpChi": args.eta,
            "reddit": args.eta,
            "Enron": args.eta,
            "wiki": args.eta,
            "BlogCatalog": args.eta,
            "Flickr": args.eta,
            "tfinance": args.eta,
        }
        dropout_dic = {
            "Amazon": args.dropout_dic,
            "Facebook": args.dropout_dic,
            "YelpChi": args.dropout_dic,
            "reddit": args.dropout_dic,
            "Enron": args.dropout_dic,
            "wiki": args.dropout_dic,
            "BlogCatalog": args.dropout_dic,
            "Flickr": args.dropout_dic,
            "tfinance": args.dropout_dic,
        }
        num_epoch_dic = {
            "Amazon": args.num_epoch,
            "Facebook": args.num_epoch,
            "YelpChi": args.num_epoch,
            "reddit": args.num_epoch,
            "Enron": args.num_epoch,
            "wiki": args.num_epoch,
            "BlogCatalog": args.num_epoch,
            "Flickr": args.num_epoch,
            "tfinance": args.num_epoch,
        }

        for encoder_name in Encoder_name_list:
            print("*" * 20, encoder_name, "*" * 20)
            print_params = {}
            print_params["final_score"] = []
            print_params["a_score"] = []
            print_params["s_score"] = []
            et = []
            for run_id in range(runs):
                seed = seed_list[run_id]
                set_seed(seed)
                if data_name in truth_list:
                    # graph = load_truth_data(data_path=data_path,dataset_name=data_name)
                    if data_name in ["Facebook", "YelpChi"]:
                        graph = load_local(data_name=data_name, raw_dir=data_path)
                    else:
                        graph = load_truth_data(
                            data_path=data_path, dataset_name=data_name
                        )
                elif data_name == "custom":
                    graph = load_custom_data(data_path=data_path)
                else:
                    graph = load_data(data_name)
                    from pathlib import Path
                    from the_utils import make_parent_dirs

                    c_p = Path(f"./tmp/inject/{data_name}.c")
                    s_p = Path(f"./tmp/inject/{data_name}.s")
                    if c_p.exists():
                        graph = torch.load(c_p)
                    else:
                        make_parent_dirs(c_p)
                        graph = inject_contextual_anomalies(
                            graph=graph, k=K, p=P, q=Q_MAP[data_name], seed=4096
                        )
                        torch.save(graph, c_p, pickle_protocol=4)
                    if s_p.exists():
                        graph = torch.load(s_p)
                    else:
                        make_parent_dirs(s_p)
                        graph = inject_structural_anomalies(
                            graph=graph, p=P, q=Q_MAP[data_name], seed=4096
                        )

                        torch.save(graph, s_p, pickle_protocol=4)
                # pca = PCA(n_components=10)
                label = graph.ndata["label"]
                features = graph.ndata["feat"]

                # graph.ndata['feat'] = F.normalize(features,dim=0)

                # pca_feat = pca.fit_transform(features.numpy())
                # graph.ndata['feat'] = torch.from_numpy(np.array(pca_feat))
                if data_name not in ["BlogCatalog", "Flickr"]:
                    mms = MinMaxScaler(feature_range=(0, 1))
                    tmp_feat = mms.fit_transform(features.numpy())
                    graph.ndata["feat"] = torch.from_numpy(np.array(tmp_feat))

                feat_size = graph.ndata["feat"].shape[1]

                # other params
                other_params = {
                    # 'runs':run_id,
                    "seed": seed,
                    "data_name": data_name,
                }
                # model params
                model_params = {
                    "in_feats": feat_size,
                    "hid_feats": hid_feats_dic[data_name],
                    "mlp_hidden_size": mlp_hidden_dic[data_name],
                    "projection_size": projection_dic[data_name],
                    "struct_dec_act": struct_dec_act_dic[data_name],
                    "dropout": dropout_dic[data_name],
                    "m": m_dic[data_name],
                    "k": k_dic[data_name],
                    "encoder_name": encoder_name,
                    "init": model_init_dic[data_name],
                }
                # fit params
                fit_params = {
                    "lr": lr_dic[data_name],
                    "weight_decay": args.weight_decay,
                    "alpha": alpha_dic[data_name],
                    "eta": eta_dic[data_name],
                    "num_epoch": num_epoch_dic[data_name],
                    "device": f"{args.gpu}",
                    "batch_size": 0,
                }

                model = CONAD(**model_params)
                # print(model)
                elapsed_time = model.fit(graph, **fit_params)
                et.append(elapsed_time)
                result = model.predict(
                    graph,
                    alpha=fit_params["alpha"],
                    device=fit_params["device"],
                    batch_size=fit_params["batch_size"],
                )
                final_score, a_score, s_score = split_auc(label, result)

                print_params["final_score"].append(final_score)
                print_params["a_score"].append(a_score)
                print_params["s_score"].append(s_score)

            elapsed_time = np.array(et)
            save_to_csv_files(
                {
                    "time": f"{elapsed_time.mean():.2f}±{elapsed_time.std():.2f}",
                },
                "time.csv",
                insert_info={
                    "model": "FAGAD",
                    "dataet": data_name,
                },
            )

            print_params["final_score"] = (
                f"{np.array(print_params['final_score']).mean()*100:.2f}±{np.array(print_params['final_score']).std()*100:.2f}"
            )
            print_params["a_score"] = (
                f"{np.array(print_params['a_score']).mean()*100:.2f}±{np.array(print_params['a_score']).std()*100:.2f}"
            )
            print_params["s_score"] = (
                f"{np.array(print_params['s_score']).mean()*100:.2f}±{np.array(print_params['s_score']).std()*100:.2f}"
            )

            if save_path != None:
                # csv2file(save_path,
                #         list(print_params.keys()),
                #         list(print_params.values())
                #         )
                print(f"write result to {save_path}")
                save_to_csv_files(
                    results=print_params,
                    insert_info={
                        "dataset": args.dataset,
                        "model": "SAG",
                    },
                    append_info={
                        "args": {
                            **other_params,
                            **model_params,
                            **fit_params,
                        },
                    },
                    csv_name="SAG.csv",
                )
            print(print_params)

# CUDA_VISIBLE_DEVICES=1 nohup python -u -m dgld.models.SaGA_v1.models > ../logs/SaGA_v1/UFA_elliptic_0301.log 2>&1 &
