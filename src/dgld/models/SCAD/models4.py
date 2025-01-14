import math
import random
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
from dgld.utils.argparser import parse_all_args
from dgld.utils.common import csv2file
from dgld.utils.common import seed_everything
from dgld.utils.common_params import K
from dgld.utils.common_params import P
from dgld.utils.common_params import Q_MAP
from dgld.utils.evaluation import split_auc
from dgld.utils.inject_anomalies import inject_contextual_anomalies
from dgld.utils.inject_anomalies import inject_structural_anomalies
from dgld.utils.load_data import load_custom_data
from dgld.utils.load_data import load_data
from dgld.utils.load_data import load_truth_data
from scipy.linalg import hadamard
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from .modules import Encoder
from .modules import HighGCN
from .modules import InnerProductDecoder
from .modules import LowGCN
from .scad_utils import *

warnings.filterwarnings("ignore")
import nni

nni_mode = True


class SCAD(nn.Module):

    def __init__(
        self,
        in_feats,
        hid_feats,
        mlp_hids,
        n_clusters,
        k,
        init="random",
        act="None",
    ):
        super(SCAD, self).__init__()
        self.n_clusters = n_clusters
        # cluster layer
        self.cluster_layer = nn.Parameter(
            torch.Tensor(self.n_clusters, hid_feats[-1]))
        self.radius = torch.tensor(
            [0] *
            n_clusters).float()  # radius R initialized with 0 by default.
        act = get_activation(act)

        self.low_gcn = LowGCN(in_feats, hid_feats, mlp_hids, act=act)
        self.high_gcn = HighGCN(in_feats, hid_feats, mlp_hids, act=act)
        self.AE = Encoder(in_feats, hid_feats[-1], mlp_hids, k, act)

        self.inner_product_decoder = InnerProductDecoder()

        for module in self.modules():
            init_weights(module, init=init)

    def fit(
        self,
        graph,
        features,
        label,
        udp=10,
        pre_epochs=100,
        pre_lr=1e-2,
        lr=1e-3,
        weight_decay=0.0,
        num_epochs=100,
        beta=0.2,
        alpha=0.5,
        lbda=0.4,
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

        pre_optimizer = torch.optim.Adam(self.low_gcn.parameters(),
                                         lr=pre_lr,
                                         weight_decay=weight_decay)
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=lr,
                                     weight_decay=weight_decay)

        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()

        self.to(device)
        graph = graph.to(device)
        features = features.to(device)
        adj = graph.adj_external(scipy_fmt="csr")
        self.lbls = torch.FloatTensor(adj.todense()).view(-1).to(device)
        # increase the recall of positive samples
        self.pos_weight = torch.FloatTensor([
            (float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum())
        ]).to(device)
        self.norm_weights = (adj.shape[0] * adj.shape[0] / float(
            (adj.shape[0] * adj.shape[0] - adj.sum()) * 2))

        # pretrain
        for ep in range(pre_epochs):
            self.train()
            z = self.low_gcn(graph, features)
            preds = self.inner_product_decoder(z).view(-1)
            loss = bce_loss(
                preds,
                self.lbls,
                norm=self.norm_weights,
                pos_weight=self.pos_weight,
            )
            pre_optimizer.zero_grad()
            loss.backward()
            pre_optimizer.step()
            if ep % 10 == 0:
                print(
                    f"Epoch:{ep}",
                    f",loss={loss.item()}",
                )

        # init cluster center
        self.eval()
        with torch.no_grad():
            low_emb = self.low_gcn(graph, features)
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        _ = kmeans.fit_predict(low_emb.data.cpu().numpy())
        self.cluster_layer.data = torch.Tensor(
            kmeans.cluster_centers_).to(device)
        self.radius = self.radius.to(device)

        if batch_size == 0:
            print("\n\nfull graph training!!!\n")
            for epoch in range(num_epochs):
                self.train()

                low_emb = self.low_gcn(graph, features)
                high_emb = self.high_gcn(graph, features)
                # kl loss
                if epoch % udp == 0:
                    self.eval()
                    with torch.no_grad():
                        low_emb = self.low_gcn(graph, features)
                        Q = self.get_Q(low_emb)

                low_q = self.get_Q(low_emb)
                high_q = self.get_Q(high_emb)

                p = target_distribution(Q.detach())
                low_kl_loss = F.kl_div(low_q.log(), p)
                high_kl_loss = F.kl_div(high_q.log(), p)

                # Hypersphere Learning
                hyper_loss, dist_to_centers = self.get_hyper_loss(
                    high_emb, beta)

                # reconstruction loss
                x_hat, a_hat = self.AE(graph, features)
                adj_orig = graph.adj_external().to_dense().to(device)
                recon_loss, struct_loss, feat_loss = self.get_rec_loss(
                    adj_orig, a_hat, features, x_hat, alpha)

                loss = (lbda * (low_kl_loss + high_kl_loss + hyper_loss) +
                        (1 - lbda) * recon_loss.mean())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新半径
                self.radius.data = torch.tensor(get_radius(
                    dist_to_centers, beta),
                                                device=device)

                self.eval()
                with torch.no_grad():
                    _, dist_score = self.getDistanceToClusters(high_emb)
                    dist_score, position = torch.min(dist_score, dim=1)

                    x_hat, a_hat = self.AE(graph, features)
                    adj_orig = graph.adj_external().to_dense().to(device)
                    recon_score, _, _ = self.get_rec_loss(
                        adj_orig, a_hat, features, x_hat, alpha)
                    predict_score = lbda * dist_score + (1 -
                                                         lbda) * recon_score
                    final_score, a_score, s_score = split_auc(
                        label,
                        predict_score.cpu().detach().numpy())

                print(
                    f"Epoch:{epoch}",
                    f",loss={loss.item()}",
                    f",low_kl_loss={low_kl_loss.item()}",
                    f",high_kl_loss={high_kl_loss.item()}",
                    f",hyper_loss={hyper_loss.item()}",
                    f",recon_loss={recon_loss.mean().item()}",
                    f",struct_loss={struct_loss.item()}",
                    f",feat_loss={feat_loss.item()}",
                    f",final_score={final_score}",
                )
                if nni_mode:
                    # 汇报中间结果
                    nni.report_intermediate_result(final_score)
            if nni_mode:
                # 汇报最终结果
                nni.report_final_result(final_score)

        else:
            print("batch graph training!!!")

    def get_rec_loss(self, a, a_hat, x, x_hat, alpha):
        diff_attribute = torch.pow(x_hat - x, 2)
        attribute_reconstruction_errors = torch.sqrt(
            torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)

        # structure reconstruction loss
        diff_structure = torch.pow(a_hat - a, 2)
        structure_reconstruction_errors = torch.sqrt(
            torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)

        cost = (alpha * attribute_reconstruction_errors +
                (1 - alpha) * structure_reconstruction_errors)

        return cost, structure_cost, attribute_cost

    def get_hyper_loss(self, x, beta):
        xe = torch.unsqueeze(x, 1) - self.cluster_layer  # [n,C,d]
        dist_to_centers = torch.sum(torch.mul(xe, xe), 2)  # [n,C]
        # euclidean_dist = torch.sqrt(dist_to_centers)
        # dist_to_centers = torch.unsqueeze(dist_to_centers,1)
        scores = dist_to_centers - self.radius**2
        loss = torch.mean(self.radius**2) + (1 / beta) * torch.mean(
            torch.max(torch.zeros_like(scores), scores))
        return loss, dist_to_centers

    def getDistanceToClusters(self, x):
        """
        obtain the distance to cluster centroids for each instance
        Args:
            x: sample on the embedded space

        Returns: square of the euclidean distance, and the euclidean distance

        """
        xe = torch.unsqueeze(x, 1) - self.cluster_layer  # [n,C,d]
        dist_to_centers = torch.sum(torch.mul(xe, xe), 2)  # [n,C]
        euclidean_dist = torch.sqrt(dist_to_centers)

        return dist_to_centers, euclidean_dist

    def get_Q(self, z):
        """get soft clustering assignment distribution

        Args:
            z (torch.Tensor): node embedding

        Returns:
            torch.Tensor: Soft assignments
        """
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / 1)
        q = q.pow((1 + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

    def predict(
        self,
        graph,
        features,
        alpha=0.5,
        lbda=0.4,
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

        self.eval()
        if batch_size == 0:
            graph = graph.remove_self_loop().add_self_loop().to(device)
            self.to(device)
            graph = graph.to(device)
            features = features.to(device)
            high_emb = self.high_gcn(graph, features)
            x_hat, a_hat = self.AE(graph, features)
            adj_orig = graph.adj_external().to_dense().to(device)
            recon_loss, struct_loss, feat_loss = self.get_rec_loss(
                adj_orig, a_hat, features, x_hat, alpha)

            _, dist_score = self.getDistanceToClusters(high_emb)
            dist_score, position = torch.min(dist_score, dim=1)
            predict_score = lbda * dist_score + (1 - lbda) * recon_loss

        else:
            pass

        return predict_score.cpu().detach().numpy()


# for test
if __name__ == "__main__":

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
    ]
    data_path = "dgld/data"
    save_path = None  #'dgld/models/SaGA_v1/results/others_Flickr_0815_1033.csv'
    runs = 1
    seed_list = [random.randint(0, 99999) for i in range(runs)]

    data_name = "BlogCatalog"

    seed = 4096
    seed_everything(seed)
    if data_name in truth_list:
        graph = load_truth_data(data_path=data_path, dataset_name=data_name)
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

    # pca = PCA(n_components=10)
    # graph.ndata['feat'] = F.normalize(features,dim=0)
    # pca_feat = pca.fit_transform(features.numpy())
    # graph.ndata['feat'] = torch.from_numpy(np.array(pca_feat))
    if data_name not in ["BlogCatalog", "Flickr"]:
        features = graph.ndata["feat"]
        mms = MinMaxScaler(feature_range=(0, 1))
        tmp_feat = mms.fit_transform(features.numpy())
        graph.ndata["feat"] = torch.from_numpy(np.array(tmp_feat))

    label = graph.ndata["label"]
    features = graph.ndata["feat"]

    feat_size = graph.ndata["feat"].shape[1]
    if nni_mode:
        tuner_params = nni.get_next_parameter()  # 这会获得一组搜索空间中的参数
        optim_params = {
            "hid_feats": [512],
            "mlp_hids": [512, 256],
            "n_clusters": 6,
            "k": 6,
            "init": "zero",
            "act": "relu",
            "udp": 5,
            "pre_epochs": 200,
            "pre_lr": 0.01,
            "lr": 0.01,
            "num_epochs": 200,
            "beta": 0.2,
            "alpha": 0.2,
            "lbda": 0.2,
        }
        optim_params.update(tuner_params)

    print_params = {}
    # other params
    other_params = {
        "seed": seed,
        "data_name": data_name,
    }
    if nni_mode:
        # model params
        model_params = {
            "in_feats": feat_size,
            "hid_feats": optim_params["hid_feats"],
            "mlp_hids": optim_params["mlp_hids"],
            "n_clusters": optim_params["n_clusters"],
            "k": optim_params["k"],
            "init": optim_params["init"],
            "act": optim_params["act"],
        }
        # fit params
        fit_params = {
            "udp": optim_params["udp"],
            "pre_epochs": optim_params["pre_epochs"],
            "pre_lr": optim_params["pre_lr"],
            "lr": optim_params["lr"],
            "num_epochs": optim_params["num_epochs"],
            "beta": optim_params["beta"],
            "alpha": optim_params["alpha"],
            "lbda": optim_params["lbda"],
            "device": "0",
            "batch_size": 0,
        }

    print(label.unique())
    model = SCAD(**model_params)
    print(model)
    model.fit(graph, features, label, **fit_params)
    result = model.predict(
        graph,
        features,
        alpha=optim_params["alpha"],
        lbda=optim_params["lbda"],
        device=fit_params["device"],
        batch_size=fit_params["batch_size"],
    )
    final_score, a_score, s_score = split_auc(label, result)

    print_params.update(other_params)
    print_params.update(model_params)
    print_params.update(fit_params)
    print_params["final_score"] = final_score
    print_params["a_score"] = a_score
    print_params["s_score"] = s_score

    if not nni_mode:
        if save_path != None:
            csv2file(save_path, list(print_params.keys()),
                     list(print_params.values()))
            print(f"write result to {save_path}")
    print(print_params)

# CUDA_VISIBLE_DEVICES=3 nohup python -u -m dgld.models.SCAD.models > ../logs/SCAD/BlogCatalog_1111.log 2>&1 &

# nnictl create --config config4.yml -p 8094
# nnictl stop --all
