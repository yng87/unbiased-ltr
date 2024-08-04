import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from data_utils import (
    get_clicks,
    get_features,
    get_group,
    get_unbiased_labels,
    sort_by_group,
)

import wandb
from wandb.integration.lightgbm import log_summary, wandb_callback

logger = logging.getLogger(__name__)


def train(
    df_train: pd.DataFrame, df_val: pd.DataFrame, model_params: dict[str, Any]
) -> lgb.LGBMRanker:
    df_train = sort_by_group(df_train)
    df_val = sort_by_group(df_val)

    X_train = get_features(df_train)
    y_train = get_clicks(df_train)
    group_train = get_group(df_train).value_counts(sort=False)

    X_val = get_features(df_val)
    y_val = get_clicks(df_val)
    group_val = get_group(df_val).value_counts(sort=False)

    wandb.config.update(model_params)
    wandb.log(
        {
            "train_data_size": X_train.shape[0],
            "num_features": X_train.shape[1],
            "train_click_mean": y_train.mean(),
            "val_data_size": X_val.shape[0],
            "val_click_mean": y_val.mean(),
        }
    )
    model = lgb.LGBMRanker(
        **model_params,
    )
    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],
        eval_group=[group_train, group_val],
        eval_metric="ndcg",
        eval_at=[5],
        callbacks=[lgb.early_stopping(5), wandb_callback()],
    )

    log_summary(model.booster_, save_model_checkpoint=False)
    return model


def predict(model: lgb.LGBMRanker, df: pd.DataFrame) -> pd.DataFrame:
    X = get_features(df)
    y = get_unbiased_labels(df)
    group = get_group(df)
    wandb.log({"pred_num_group": len(np.unique(group))})
    ypred = model.predict(X)
    return pd.DataFrame({"group": group, "label": y, "prediction": ypred})
