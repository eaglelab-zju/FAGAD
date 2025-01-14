# -*- coding: utf-8 -*-
"""
Metrics used to evaluate the outlier detection performance
"""
# Author: Yingtong Dou <ytongdou@gmail.com>, Kay Liu <zliu234@uic.edu>
# License: BSD 2 clause
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def eval_roc_auc(label, score):
    """
    ROC-AUC score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    roc_auc : float
        Average ROC-AUC score across different labels.
    """

    roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc


def eval_recall_at_k(label, score, k=None):
    """
    Recall score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        recall. Default: ``None``.

    Returns
    -------
    recall_at_k : float
        Recall for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    recall_at_k = sum(label[score.topk(k).indices]) / sum(label)
    return recall_at_k


def eval_precision_at_k(label, score, k=None):
    """
    Precision score for top k instances with the highest outlier scores.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.
    k : int, optional
        The number of instances to evaluate. ``None`` for
        precision. Default: ``None``.

    Returns
    -------
    precision_at_k : float
        Precision for top k instances with the highest outlier scores.
    """

    if k is None:
        k = sum(label)
    precision_at_k = sum(label[score.topk(k).indices]) / k
    return precision_at_k


def eval_average_precision(label, score):
    """
    Average precision score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    score : torch.Tensor
        Outlier scores in shape of ``(N, )``.

    Returns
    -------
    ap : float
        Average precision score.
    """

    ap = average_precision_score(y_true=label, y_score=score)
    return ap


def eval_f1(label, pred):
    """
    F1 score for binary classification.

    Parameters
    ----------
    label : torch.Tensor
        Labels in shape of ``(N, )``, where 1 represents outliers,
        0 represents normal instances.
    pred : torch.Tensor
        Outlier prediction in shape of ``(N, )``.

    Returns
    -------
    f1 : float
        F1 score.
    """

    f1 = f1_score(y_true=label, y_pred=pred)
    return f1


import numpy as np


def split_auc(groundtruth, prob):
    """
    print the scoring(AUC) of the two types of anomalies separately and global auc.

    Parameters
    ----------
    groundtruth: np.ndarray
        Indicates whether this node is an injected anomaly node.
        0: normal node; 1: structural anomaly; 2: contextual anomaly

    prob: np.ndarray-like array
        saving the predicted score for every node

    Returns
    -------
    None
    """
    final_score = -1
    s_score = -1
    a_score = -1
    try:
        str_pos_idx = groundtruth == 1
        attr_pos_idx = groundtruth == 2
        norm_idx = groundtruth == 0

        str_data_idx = str_pos_idx | norm_idx
        attr_data_idx = attr_pos_idx | norm_idx

        str_data_groundtruth = groundtruth[str_data_idx]
        str_data_predict = prob[str_data_idx]
        attr_data_groundtruth = np.where(groundtruth[attr_data_idx] != 0, 1, 0)
        attr_data_predict = prob[attr_data_idx]

        s_score = roc_auc_score(str_data_groundtruth, str_data_predict)
        a_score = roc_auc_score(attr_data_groundtruth, attr_data_predict)
        print("structural anomaly score:", s_score)
        print("attribute anomaly score:", a_score)

        final_score = roc_auc_score(np.where(groundtruth == 0, 0, 1), prob)

    except ValueError:
        pass

    # for truth label
    if final_score == -1 and attr_pos_idx.sum() == 0 and str_pos_idx.sum(
    ) != 0:
        final_score = roc_auc_score(groundtruth, prob)

    print("final anomaly score:", final_score)
    return final_score, a_score, s_score
