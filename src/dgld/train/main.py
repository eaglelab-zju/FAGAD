import os
import random
import sys

import numpy as np
from dgld.models import *
from dgld.utils.argparser import parse_all_args
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
from dgld.utils.log import Dgldlog
from the_utils import set_device
from the_utils import set_seed
from the_utils import tab_printer
from torch_geometric.utils import from_dgl

# truth_list = ['weibo','tfinance','tsocial','reddit','Amazon','Class','Disney','elliptic','Enron']
if __name__ == "__main__":
    args_dict, args = parse_all_args()
    data_name = args_dict["dataset"]
    save_path = args.save_path
    exp_name = args.exp_name
    # log = Dgldlog(save_path,exp_name,args)

    res_list_final = []
    res_list_attrb = []
    res_list_struct = []

    device = set_device(args.device)

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
            graph = load_truth_data(data_path=data_path, dataset_name=data_name)
    elif data_name == "custom":
        graph = load_custom_data(data_path=data_path)
    else:
        graph = load_data(data_name)
        graph = inject_contextual_anomalies(
            graph=graph, k=K, p=P, q=Q_MAP[data_name], seed=4096
        )
        graph = inject_structural_anomalies(
            graph=graph, p=P, q=Q_MAP[data_name], seed=4096
        )
        if data_name not in ["BlogCatalog", "Flickr"]:
            from sklearn.preprocessing import LabelEncoder, MinMaxScaler

            mms = MinMaxScaler(feature_range=(0, 1))
            tmp_feat = mms.fit_transform(graph.ndata["feat"].numpy())
            graph.ndata["feat"] = torch.from_numpy(np.array(tmp_feat))
    label = graph.ndata["label"]
    graph.name = data_name
    # data = from_dgl(graph)
    # data.name = data_name
    # data.num_classes = 2
    # data.x = data.feat
    # ata.y = data.label
    # data.num_nodes = graph.num_nodes()
    # data.num_edges = graph.num_edges()
    # data.edge_index = torch.stack(graph.edges(), dim=0)
    # data = data

    seed_list = [random.randint(0, 99999) for i in range(args.runs)]
    seed_list[0] = args_dict["seed"]
    if "feat_size" in args_dict["model"]:
        args_dict["model"]["feat_size"] = graph.ndata["feat"].shape[1]
    if "num_feats" in args_dict["model"]:
        args_dict["model"]["num_feats"] = graph.ndata["feat"].shape[1]
    if "in_feats" in args_dict["model"]:
        args_dict["model"]["in_feats"] = graph.ndata["feat"].shape[1]
    if "attrb_dim" in args_dict["model"]:
        args_dict["model"]["attrb_dim"] = graph.ndata["feat"].shape[1]
    if "num_nodes" in args_dict["model"]:
        args_dict["model"]["num_nodes"] = graph.num_nodes()
    if "args" in args_dict["model"]:
        args_dict["model"]["args"] = args
    tab_printer(args_dict)
    et = []
    for runs in range(args.runs):
        # log.update_runs()
        seed = seed_list[runs]
        # seed_everything(seed)
        set_seed(seed)
        args_dict["seed"] = seed

        # if data_name in truth_list:
        #     graph = load_truth_data(data_path=args.data_path,dataset_name=data_name)
        # elif data_name == 'custom':
        #     graph = load_custom_data(data_path=args.data_path)
        # else:
        #     graph = load_data(data_name)
        #     graph = inject_contextual_anomalies(graph=graph,k=K,p=P,q=Q_MAP[data_name],seed=seed)
        #     graph = inject_structural_anomalies(graph=graph,p=P,q=Q_MAP[data_name],seed=seed)

        # label = graph.ndata['label']

        if args.model in [
            "DOMINANT",
            "AnomalyDAE",
            "ComGA",
            "DONE",
            "AdONE",
            "CONAD",
            "ALARM",
            "ONE",
            "GAAN",
            "GUIDE",
            "CoLA",
            "AAGNN",
            "SLGAD",
            "ANEMONE",
            "GCNAE",
            "MLPAE",
            "SCAN",
        ]:
            model = eval(f'{args.model}(**args_dict["model"])')
        else:
            raise ValueError(f"{args.model} is not implemented!")

        t = model.fit(graph, **args_dict["fit"])
        result = model.predict(graph, **args_dict["predict"])
        auc, a_score, s_score = split_auc(label, result)
        res_list_final.append(auc)
        res_list_attrb.append(a_score)
        res_list_struct.append(s_score)
        et.append(t)

    import numpy as np
    from the_utils import save_to_csv_files

    elapsed_time = np.array(et)
    save_to_csv_files(
        {
            "time": f"{elapsed_time.mean():.2f}±{elapsed_time.std():.2f}",
        },
        "time.csv",
        insert_info={
            "model": args.model,
            "dataet": data_name,
        },
    )

    from the_utils import save_to_csv_files

    save_path = "baselines.csv"
    print(f"write result to {save_path}")
    save_to_csv_files(
        results={
            "AUC": f"{np.array(res_list_final).mean()*100:.2f}±{np.array(res_list_final).std()*100:.2f}",
        },
        insert_info={
            "dataset": args.dataset,
            "model": model.__class__.__name__,
        },
        append_info={
            "args": args_dict,
        },
        csv_name=save_path,
    )
