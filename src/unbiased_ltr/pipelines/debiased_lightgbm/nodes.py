import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from data_utils import (
    get_clicks,
    get_features,
    get_group,
    get_position,
    get_unbiased_labels,
    sort_by_group,
)

import wandb
from wandb.integration.lightgbm import log_summary, wandb_callback

logger = logging.getLogger(__name__)


def train(
    df_train: pd.DataFrame, df_val: pd.DataFrame, model_params: dict[str, Any]
) -> lgb.Booster:
    df_train = sort_by_group(df_train)
    df_val = sort_by_group(df_val)

    X_train = get_features(df_train)
    y_train = get_clicks(df_train)
    group_train = get_group(df_train).value_counts(sort=False)
    position_train = get_position(df_train)

    X_val = get_features(df_val)
    y_val = get_clicks(df_val)
    group_val = get_group(df_val).value_counts(sort=False)
    position_val = np.ones(X_val.shape[0])

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

    ds_train = lgb.Dataset(
        X_train,
        label=y_train,
        group=group_train,
        position=position_train,
    )
    ds_val = lgb.Dataset(
        X_val,
        label=y_val,
        group=group_val,
        position=position_val,
    )

    model_params.update(
        {
            "metric": "ndcg",
            "ndcg_at": [5],
        }
    )

    model = lgb.train(
        model_params,
        ds_train,
        valid_sets=[ds_train, ds_val],
        valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(-1), lgb.early_stopping(5), wandb_callback()],
    )

    log_summary(model, save_model_checkpoint=False)
    return model


def predict(model: lgb.Booster, df: pd.DataFrame) -> pd.DataFrame:
    X = get_features(df)
    y = get_unbiased_labels(df)
    group = get_group(df)
    position = np.ones(X.shape[0])
    wandb.log({"pred_num_group": len(np.unique(group))})
    ypred = model.predict(X, num_iteration=model.best_iteration)
    return pd.DataFrame({"group": group, "label": y, "prediction": ypred})
