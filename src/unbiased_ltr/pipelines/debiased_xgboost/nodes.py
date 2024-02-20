import logging
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from data_utils import (
    get_clicks,
    get_features,
    get_group,
    get_position,
    get_unbiased_labels,
    sort_by_group,
)

import wandb
from wandb.xgboost import WandbCallback

logger = logging.getLogger(__name__)


def train(
    df_train: pd.DataFrame, df_val: pd.DataFrame, model_params: dict[str, Any]
) -> xgb.XGBRanker:
    df_train = sort_by_group(df_train)
    df_val = sort_by_group(df_val)

    X_train = get_features(df_train)
    X_train["position"] = get_position(df_train)
    y_train = get_clicks(df_train)
    group_train = get_group(df_train)

    X_val = get_features(df_val)
    X_val["position"] = 1
    y_val = get_clicks(df_val)
    group_val = get_group(df_val)

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
    model = xgb.XGBRanker(
        callbacks=[
            WandbCallback(log_model=False, log_feature_importance=True),
        ],
        **model_params,
    )
    model.fit(
        X_train,
        y_train,
        qid=group_train,
        eval_set=[
            (X_train, y_train),
            (X_val, y_val),
        ],
        eval_qid=[group_train, group_val],
        verbose=10,
    )
    return model


def predict(model: xgb.XGBRanker, df: pd.DataFrame) -> pd.DataFrame:
    X = get_features(df)
    X["position"] = 1
    y = get_unbiased_labels(df)
    group = get_group(df)
    wandb.log({"pred_num_group": len(np.unique(group))})
    ypred = model.predict(X)
    return pd.DataFrame({"group": group, "label": y, "prediction": ypred})
