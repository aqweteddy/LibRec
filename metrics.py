from os import pread
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

def mrr(pred, ground):
    """mrr

    Args:
        pred (List[int]): [B, candi]
        ground (List[int]): [description]

    Returns:
        [type]: [description]
    """
    s = 0.
    for i, p in enumerate(pred):
        if ground[p]:
            s += 1 / (i+1)
    return s / len(ground)
    # return ground[pred[0]]

def roc_auc(pred, ground):
    return roc_auc_score(ground, pred,multi_class='ovo')