import numpy as np


def dcg(labels: np.ndarray, preds: np.ndarray, k: int) -> float:
    sorted_idx = np.argsort(-preds)[:k]
    sorted_labels = labels[sorted_idx]
    length = min(k, len(labels))
    scores = (2 ** (sorted_labels) - 1) / np.log2(np.arange(length) + 2)
    return scores.sum()


def idcg(labels: np.ndarray, k: int) -> float:
    return dcg(labels, labels, k)


def ndcg(labels: np.ndarray, preds: np.ndarray, k: int) -> float:
    assert k > 0
    assert len(labels) > 0
    assert len(preds) > 0
    assert len(labels) == len(preds)
    if labels.sum() == 0:
        # 評価する意味がないクエリが時々ある
        return 0.0
    return dcg(labels, preds, k) / idcg(labels, k)


def calc_mean_ndcg(
    group_boundary_indices: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    k: int,
) -> np.floating:
    ndcgs = []
    for i in range(len(group_boundary_indices) - 1):
        idx_from = group_boundary_indices[i]
        idx_to = group_boundary_indices[i + 1]
        ndcgs.append(ndcg(labels[idx_from:idx_to], preds[idx_from:idx_to], k))
    return np.mean(ndcgs)
