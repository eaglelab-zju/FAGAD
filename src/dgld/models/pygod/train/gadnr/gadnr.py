import argparse

import numpy as np
from dgld.models.pygod.detector import GADNR
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_auc_score
from the_utils import save_to_csv_files
from the_utils import set_device
from the_utils import set_seed
from the_utils import tab_printer
from torch_geometric.utils import from_dgl

# pylint: disable=pointless-string-statement
"""
    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in the backbone encoder model. Default: ``1``.
    deg_dec_layers : int, optional
        The number of layers for the node degree decoder. Default: ``4``.
    fea_dec_layers : int, optional
        The number of layers for the node feature decoder. Default: ``3``.
    backbone : torch.nn.Module, optional
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    sample_size : int, optional
        The number of samples for the neighborhood distribution.
        Default: ``2``.
    sample_time : int, optional
        The number sample times to remove the noise during node feature and
        neighborhood distribution reconstruction. Default: ``3``.
    neigh_loss : str, optional
        The neighbor reconstruction loss. ``KL`` represents the KL divergence
        loss, ``W2`` represents the W2 loss. Default: ``KL``.
    lambda_loss1 : float, optional
        The weight of the neighborhood reconstruction loss term.
        Default: ``1e-2``.
    lambda_loss2 : float, optional
        The weight of the node feature reconstruction loss term.
        Default: ``1e-3``.
    lambda_loss3 : float, optional
        The weight of the node degree reconstruction loss term.
        Default: ``1e-4``.
    real_loss : bool, optional
        Whether using the original loss proposed in the paper as the
        decision score, if not, using the proposed weighted decision score.
        Default: ``True``.
    lr : float, optional
        Learning rate. Default: ``0.01``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.0003``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GADNR")
    parser.add_argument("--dataset", type=str, default="Amazon")
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hid_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--fea_dec_layers", type=int, default=3)
    parser.add_argument("--nni", action="store_true", default=False)

    args = parser.parse_args()
    try:
        import nni

        NNI = args.nni
    except Exception as e:
        print("no NNI")
        NNI = False

    params = {
        "hid_dim": 64,
        "lr": 0.001,
        "dropout": 0.2,
        "num_layers": 1,
        "fea_dec_layers": 2,
        **args.__dict__,
    }
    optimized_params = nni.get_next_parameter() if NNI else {}
    params.update(optimized_params)

    device = set_device(args.gpu)

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

    data = from_dgl(graph)
    data.name = data_name
    data.num_classes = 2
    data.x = data.feat
    data.y = data.label
    data.num_nodes = graph.num_nodes()
    data.num_edges = graph.num_edges()
    data.edge_index = torch.stack(graph.edges(), dim=0)
    data = data.to(device)

    tab_printer(params)

    runs = 3
    import random

    seed_list = [random.randint(0, 9999999) for i in range(runs)]
    aucs = []
    for run in range(runs):
        set_seed(seed_list[run])
        model = GADNR(
            hid_dim=params["hid_dim"],
            num_layers=params["num_layers"],
            deg_dec_layers=4,
            fea_dec_layers=params["fea_dec_layers"],
            sample_size=2,
            sample_time=3,
            neigh_loss="KL",
            lambda_loss1=1e-2,
            lambda_loss2=1e-1,
            lambda_loss3=8e-1,
            real_loss=True,
            lr=params["lr"],
            epoch=100,
            dropout=params["dropout"],
            weight_decay=params["weight_decay"],
            gpu=params["gpu"],
            batch_size=0,
            num_neigh=-1,
            contamination=0.1,
            verbose=2,
        )

        model.fit(data=data, label=data.y.cpu(), NNI=NNI)
        # get outlier scores on the training data (transductive setting)
        score = model.decision_score_

        # predict labels and scores on the testing data (inductive setting)
        # pred, score = model.predict(data, return_score=True)

        from dgld.utils.evaluation import split_auc

        auc, a_score, s_score = split_auc(data.y.cpu(), score)
        aucs.append(auc)
    if NNI:
        nni.report_final_result(np.array(aucs).mean())
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
        append_info={"args": params},
        csv_name=save_path,
    )

    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_flickr.yaml --port 8881 > logs/gadnr_flickr.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_blog.yaml --port 8882 > logs/gadnr_blog.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_red.yaml --port 8884 > logs/gadnr_red.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_wiki.yaml --port 8885 > logs/gadnr_wiki.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_enr.yaml --port 8886 > logs/gadnr_enr.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_fb.yaml --port 8887 > logs/gadnr_fb.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_ama.yaml --port 8888 > logs/gadnr_ama.log &
    # nohup nnictl create --config dgld/models/pygod/train/gadnr/gadnr_yelp.yaml --port 8889 > logs/gadnr_yelp.log &
    # NOTE: `.env/lib/python3.8/site-packages/nni/experiment/launcher.py` modify `_check_rest_server(port, url_prefix=url_prefix)` into `_check_rest_server(port, url_prefix=url_prefix, retry=20)`
