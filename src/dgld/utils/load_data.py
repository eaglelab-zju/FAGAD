import os
import ssl
import sys
from typing import *

import dgl
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
import wget
from ogb.nodeproppred import DglNodePropPredDataset

from .common import is_bidirected
from .common import preprocess_features

current_file_name = __file__
current_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(current_file_name)))
data_path = current_dir + "/data/"

dgl_datasets = ["Cora", "Citeseer", "Pubmed"]
ogb_datasets = ["ogbn-arxiv"]
other_datasets = ["BlogCatalog", "Flickr", "YelpChi", "Facebook"]

from the_utils import tab_printer


def print_dataset_info(
    dataset_name: str,
    n_nodes: int,
    n_edges: int,
    n_feats: int,
    n_clusters: int,
    self_loops: int = None,
    is_directed: bool = None,
    thead: List[str] = None,
) -> None:
    dic = {
        "NumNodes": n_nodes,
        "NumEdges": n_edges,
        "NumFeats": n_feats,
        "NumClasses": n_clusters,
        "Self-loops": self_loops,
        "Directed": is_directed,
    }

    if self_loops is None:
        dic.pop("Self-loops")
    if is_directed is None:
        dic.pop("Directed")

    tab_printer(
        dic,
        thead=["Dataset", dataset_name] if thead is None else thead,
        cols_align=["c", "r"],
        sort=False,
    )


def load_data(
    dataset_name=None,
    raw_dir=data_path,
    feat_norm=True,
    add_self_loop=False,
    init_label=True,
):
    """
    load data

    Parameters
    ----------
    dataset_name : str
        name of dataset
    data_path : str
        the file to read in
    feat_norm : bool, optional
        process features, here norm in row, by default True
    add_self_loop : bool, optional
        if add self loop to graph, by default False

    Returns
    -------
    graph : DGL.graph
        the graph read from data_path,default row feature norm.
    """
    if dataset_name in dgl_datasets:
        graph = getattr(dgl.data,
                        dataset_name + "GraphDataset")(raw_dir=raw_dir)[0]
    elif dataset_name in ogb_datasets:
        graph = load_ogbn_arxiv(raw_dir=raw_dir)
    elif dataset_name in other_datasets:
        graph = other_datasets_map[dataset_name](raw_dir=raw_dir)
    else:
        raise ValueError(f"{dataset_name} dataset is not implemented!")

    assert is_bidirected(graph) == True
    # init label
    if init_label:
        graph.ndata["label"] = torch.zeros(graph.num_nodes())

    if feat_norm:
        graph.ndata["feat"] = preprocess_features(graph.ndata["feat"])

    if add_self_loop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    print_dataset_info(
        dataset_name=
        f"{dataset_name.upper()} undirected {dataset_name}\n  add_self_loop={add_self_loop}",
        n_nodes=graph.num_nodes(),
        n_edges=graph.num_edges(),
        n_feats=graph.ndata["feat"].shape[1],
        n_clusters=2,
    )

    return graph


def load_truth_data(data_path,
                    dataset_name,
                    feat_norm=False,
                    add_self_loop=False):
    """
    load truth data

    Parameters
    ----------
    data_path : str
        data path
    dataset_name: str
        dataset name
    feat_norm : bool, optional
        process features, here norm in row, by default True
    add_self_loop : bool, optional
        if add self loop to graph, by default False

    Returns
    -------
    graph : DGL.graph
        the graph read from data_path,default row feature norm.
    """
    graph = dgl.load_graphs(data_path + "/" + dataset_name)[0][0]
    if not is_bidirected(graph):
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    if "feat" in graph.ndata:
        graph.ndata["feat"] = torch.tensor(graph.ndata["feat"],
                                           dtype=torch.float32)
    else:
        graph.ndata["feat"] = graph.ndata["feature"].to(torch.float32)
        graph.ndata["label"] = torch.argmax(graph.ndata["label"], dim=-1)
        del graph.ndata["feature"]
    if feat_norm:
        graph.ndata["feat"] = preprocess_features(graph.ndata["feat"])
    if add_self_loop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")
    print_dataset_info(
        dataset_name=
        f"{dataset_name.upper()} undirected {dataset_name}\n  add_self_loop={add_self_loop}",
        n_nodes=graph.num_nodes(),
        n_edges=graph.num_edges(),
        n_feats=graph.ndata["feat"].shape[1],
        n_clusters=2,
    )
    return graph


def load_custom_data(data_path, feat_norm=True, add_self_loop=False):
    """
    load custom data

    Parameters
    ----------
    data_path : str
        data path
    feat_norm : bool, optional
        process features, here norm in row, by default True
    add_self_loop : bool, optional
        if add self loop to graph, by default False

    Returns
    -------
    graph : DGL.graph
        the graph read from data_path,default row feature norm.
    """
    # biuld graph
    edges = pd.read_csv(data_path + "edges.csv")
    nodes = pd.read_csv(data_path + "nodes.csv")
    feats = np.ascontiguousarray(
        nodes[[name for name in nodes.columns.values if "emb" in name]])
    nodes = nodes.reset_index().rename(columns={"index": "tmp_id"})
    id_old2new_dict = dict(zip(nodes.id.tolist(), nodes.tmp_id.tolist()))
    u, v = edges.id_src.map(id_old2new_dict).tolist(), edges.id_dst.map(
        id_old2new_dict)
    graph = dgl.graph((u, v))
    graph.ndata["feat"] = torch.tensor(feats, dtype=torch.float32)
    graph.ndata["label"] = torch.zeros(graph.num_nodes())

    if not is_bidirected(graph):
        graph = dgl.to_bidirected(graph, copy_ndata=True)

    if feat_norm:
        graph.ndata["feat"] = preprocess_features(graph.ndata["feat"])

    if add_self_loop:
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    return graph


def load_mat_data2dgl(data_path, verbose=True):
    """
    load data from .mat file

    Parameters
    ----------
    data_path : str
        the file to read in
    verbose : bool, optional
        print info, by default True

    Returns
    -------
    graph : DGL.graph
        the graph read from data_path
    """
    mat_path = data_path
    data_mat = sio.loadmat(mat_path)
    adj = data_mat["Network"]
    feat = data_mat["Attributes"]
    # feat = preprocessing.normalize(feat, axis=0)
    truth = data_mat["Label"]
    truth = truth.flatten()
    graph = dgl.from_scipy(adj)
    graph.ndata["feat"] = torch.from_numpy(feat.toarray()).to(torch.float32)
    graph.ndata["label"] = torch.from_numpy(truth).to(torch.float32)
    num_classes = len(np.unique(truth))

    if verbose:
        print()
        print("  DGL dataset")
        print("  NumNodes: {}".format(graph.number_of_nodes()))
        print("  NumEdges: {}".format(graph.number_of_edges()))
        print("  NumFeats: {}".format(graph.ndata["feat"].shape[1]))
        print("  NumClasses: {}".format(num_classes))

    return graph


def load_ogbn_arxiv(raw_dir=data_path):
    """
    Read ogbn-arxiv from dgl.

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.

    returns
    -------
    graph : dgl.graph
        the graph of ogbn-arxiv
    """
    data = DglNodePropPredDataset(name="ogbn-arxiv", root=raw_dir)
    graph, _ = data[0]
    graph = dgl.to_bidirected(graph, copy_ndata=True)

    return graph


# create this bar_progress method which is invoked automatically from wget
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (
        current / total * 100,
        current,
        total,
    )
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def load_BlogCatalog(raw_dir=data_path):
    """
    load BlogCatalog dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.

    Returns
    -------
    graph : DGL.graph
    Examples
    -------
    >>> graph=load_BlogCatalog()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir, "BlogCatalog.mat")
    if not os.path.exists(data_file):
        url = "https://github.com/GRAND-Lab/CoLA/blob/main/raw_dataset/BlogCatalog/BlogCatalog.mat?raw=true"
        wget.download(url, out=data_file, bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)


def load_Flickr(raw_dir=data_path):
    """
    load Flickr dgl graph

    Parameters
    ----------
    raw_dir : str
        Data path. Supports user customization.

    Returns
    -------
    graph : DGL.graph

    Examples
    -------
    >>> graph=load_Flickr()[0]
    """
    ssl._create_default_https_context = ssl._create_unverified_context
    data_file = os.path.join(raw_dir, "Flickr.mat")
    if not os.path.exists(data_file):
        url = "https://github.com/GRAND-Lab/CoLA/blob/main/raw_dataset/Flickr/Flickr.mat?raw=true"
        wget.download(url, out=data_file, bar=bar_progress)

    return load_mat_data2dgl(data_path=data_file)


def load_local(data_name, raw_dir=data_path):
    data_file = os.path.join(raw_dir, f"{data_name}.mat")
    return load_mat_data2dgl(data_path=data_file)


other_datasets_map = {
    "Flickr": load_Flickr,
    "BlogCatalog": load_BlogCatalog,
}

# for test
if __name__ == "__main__":
    import pandas as pd
    import scipy.sparse as sp
    from .inject_anomalies import (
        inject_contextual_anomalies,
        inject_structural_anomalies,
    )

    torch.set_printoptions(precision=10, profile="full")
    dataset_list = [
        "BlogCatalog",
        "Flickr",
        # "weibo",
        "tfinance",
        # "tsocial",
        "reddit",
        "Amazon",
        # "Class",
        # "Disney",
        "Facebook",
        "YelpChi",
        "elliptic",
        "Enron",
        "wiki",
        # "Cora",
        # "Citeseer",
        # "Pubmed",
        # "ogbn-arxiv",
    ]

    q_map = {
        "BlogCatalog": 10,
        "Flickr": 15,
        "Cora": 5,
        "Citeseer": 5,
        "Pubmed": 20,
        "ogbn-arxiv": 200,
    }
    from dgld.utils.common_params import Q_MAP, K, P

    n_nodes_list, n_edges_list, n_attr_list, n_anom_list = [], [], [], []
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
    types = []
    for data_name in dataset_list:
        try:
            print("\n\n\n\n", "-" * 10, data_name, "-" * 10)
            if data_name in truth_list:
                if data_name in ["Facebook", "YelpChi"]:
                    graph = load_local(data_name=data_name, raw_dir=data_path)
                else:
                    graph = load_truth_data(data_path=data_path,
                                            dataset_name=data_name)
                types.append("R")
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
                print(sum(graph.ndata["feat"][0]))
                types.append("I")
                # graph = inject_contextual_anomalies(graph=graph, k=50, p=15, q=q_map[data_name])
                # graph = inject_structural_anomalies(graph=graph, p=15, q=q_map[data_name])

            n_nodes_list.append(graph.number_of_nodes())
            n_edges_list.append(graph.number_of_edges())
            n_attr_list.append(graph.ndata["feat"].shape[1])
            n_anom_list.append(sum(graph.ndata["label"] != 0).item())
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"{data_name} not supported\n\n\n")
            continue

    df_dataset = pd.DataFrame({
        "Datasets":
        dataset_list,
        "Types":
        types,
        "Nodes":
        n_nodes_list,
        "Edges":
        n_edges_list,
        "Attributes":
        n_attr_list,
        "Anomalies":
        n_anom_list,
        "Anomalies_rate":
        [n_a / n_nodes_list[idx] for idx, n_a in enumerate(n_anom_list)],
    })
    print(df_dataset)
