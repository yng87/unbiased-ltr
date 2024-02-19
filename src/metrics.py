import numpy as np
import pandas as pd
import torch
from torchmetrics import Metric
from torchmetrics.utilities import dim_zero_cat


def dcg(relevances: np.ndarray, k: int) -> float:
    relevances = relevances[:k]
    return float(
        np.sum((np.power(2, relevances) - 1) / np.log2(np.arange(len(relevances)) + 2))
    )


def ndcg(relevances: np.ndarray, k: int) -> float:
    """Calculate NDCG@k
    Args:
        relevances: array of ground truth relevance scores
        k: cutoff of relevances
    Returns:
        NDCG@k
    """
    assert k > 0
    assert len(relevances) > 0
    dcg_at_k = dcg(relevances, k)
    ideal_ranking_idx = np.argsort(-relevances)
    idcg_at_k = dcg(relevances[ideal_ranking_idx], k)
    if idcg_at_k == 0:
        return 0.0
    return dcg_at_k / idcg_at_k


def calc_mean_ndcg(df_pred: pd.DataFrame, k: int) -> float:
    grouped = df_pred.groupby("group")
    ndcg_list = []
    for _, group_df in grouped:
        labels = group_df["label"].to_numpy()
        preds = group_df["prediction"].to_numpy()
        relevances = labels[np.argsort(-preds)]
        ndcg_list.append(ndcg(relevances, k))

    mean_ndcg = float(np.mean(ndcg_list))
    return mean_ndcg


class NDCG(Metric):
    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = k
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("labels", default=[], dist_reduce_fx="cat")
        self.add_state("groups", default=[], dist_reduce_fx="cat")

    def update(
        self, preds: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor
    ) -> None:
        self.preds.append(preds.detach().cpu().reshape(-1))
        self.labels.append(labels.detach().cpu().reshape(-1))
        self.groups.append(groups.detach().cpu().reshape(-1))

    def compute(self) -> torch.Tensor:

        preds = dim_zero_cat(self.preds).numpy()
        labels = dim_zero_cat(self.labels).numpy()
        groups = dim_zero_cat(self.groups).numpy()
        df = pd.DataFrame(
            {
                "group": groups,
                "label": labels,
                "prediction": preds,
            }
        )
        return torch.tensor(calc_mean_ndcg(df, self.k))
