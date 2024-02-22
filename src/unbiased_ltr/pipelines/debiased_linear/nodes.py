import logging
from typing import Any

import numpy as np
import pandas as pd
from data_utils import (
    get_clicks,
    get_features,
    get_group,
    get_position,
    get_unbiased_labels,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import wandb

logger = logging.getLogger(__name__)


def train(df: pd.DataFrame, model_params: dict[str, Any]) -> Pipeline:
    X = get_features(df)
    X["position"] = get_position(df)
    y = get_clicks(df)
    wandb.config.update(model_params)
    wandb.log(
        {
            "train_data_size": X.shape[0],
            "num_features": X.shape[1],
            "train_click_mean": y.mean(),
        }
    )
    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(n_jobs=-1, verbose=1, **model_params),
            ),
        ]
    )
    pipe.fit(X, y)
    wandb.sklearn.plot_feature_importances(pipe["model"], X.columns)
    return pipe


def predict(model: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    X = get_features(df)
    X["position"] = 1
    y = get_unbiased_labels(df)
    group = get_group(df)
    wandb.log({"pred_num_group": len(np.unique(group))})
    ypred = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"group": group, "label": y, "prediction": ypred})
