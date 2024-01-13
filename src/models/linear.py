import os
import pickle
from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .base import ModelBase


class LinearModel(ModelBase):
    def __init__(
        self,
        penalty: Literal["l1", "l2", "elasticnet", None] = "l2",
        C: float = 1.0,
        max_iter: int = 100,
        solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = "liblinear",
        l1_ratio: float | None = None,
        random_state: int = 42,
    ):
        self.penalty: Literal["l1", "l2", "elasticnet", None] = penalty
        self.C = C
        self.max_iter = max_iter
        self.solver: Literal[
            "lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"
        ] = solver
        self.l1_ratio = l1_ratio
        self.random_state = random_state

    def build(self):
        self.pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty=self.penalty,
                        C=self.C,
                        max_iter=self.max_iter,
                        solver=self.solver,
                        n_jobs=-1,
                        l1_ratio=self.l1_ratio,
                        random_state=self.random_state,
                        verbose=0,
                    ),
                ),
            ]
        )

    def fit(self, train_dataset, eval_dataset):
        X = train_dataset.features
        y = train_dataset.clicks
        self.pipe.fit(X, y)

    def predict(self, eval_dataset):
        X = eval_dataset.features
        return self.pipe.predict_proba(X)[:, 1]

    def save(self, output_dir: str):
        path = os.path.join(output_dir, "model.pkl")
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_dir: str):
        path = os.path.join(model_dir, "model.pkl")
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


class PositionDebiasedLinearModel(LinearModel):
    def fit(self, train_dataset, eval_dataset):
        X = np.concatenate(
            (train_dataset.features, train_dataset.positions.reshape(-1, 1)),
            axis=1,
        )
        y = train_dataset.clicks
        self.pipe.fit(X, y)

    def predict(self, eval_dataset):
        positions = np.ones(shape=(len(eval_dataset), 1))
        X = np.concatenate((eval_dataset.features, positions), axis=1)
        return self.pipe.predict_proba(X)[:, 1]
