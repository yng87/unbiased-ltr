import numpy as np
import xgboost as xgb

from .base import ModelBase


class XGBoostModel(ModelBase):
    def __init__(self):
        pass

    def build(self):
        self.model = xgb.XGBRanker(
            tree_method="hist",
            n_estimators=100,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=10,  # slightly higher than 5
            objective="rank:ndcg",
            eval_metric=["ndcg@5"],
            early_stopping_rounds=5,
            random_state=42,
        )

    def fit(self, train_dataset, eval_dataset):
        features_train = train_dataset.features
        clicks_train = train_dataset.clicks
        qid_train = train_dataset.queries

        features_eval = eval_dataset.features
        clicks_eval = eval_dataset.clicks
        qid_eval = eval_dataset.queries

        self.model.fit(
            features_train,
            clicks_train,
            qid=qid_train,
            eval_set=[
                (features_train, clicks_train),
                (features_eval, clicks_eval),
            ],
            eval_qid=[qid_train, qid_eval],
            verbose=10,
        )

    def predict(self, eval_dataset):
        return self.model.predict(eval_dataset.features)


class PositionDebiasedXGBoostModel(ModelBase):
    def __init__(self):
        pass

    def build(self):
        self.model = xgb.XGBRanker(
            tree_method="hist",
            n_estimators=100,
            lambdarank_pair_method="topk",
            lambdarank_num_pair_per_sample=10,  # slightly higher than 5
            objective="rank:ndcg",
            eval_metric=["ndcg@5"],
            early_stopping_rounds=5,
            random_state=42,
        )

    def fit(self, train_dataset, eval_dataset):
        pos_train = train_dataset.positions.reshape(-1, 1)
        X_train = np.concatenate((train_dataset.features, pos_train), axis=1)
        clicks_train = train_dataset.clicks
        qid_train = train_dataset.queries

        pos_eval = np.ones((len(eval_dataset), 1), dtype=np.float32)
        X_eval = np.concatenate((eval_dataset.features, pos_eval), axis=1)
        clicks_eval = eval_dataset.clicks
        qid_eval = eval_dataset.queries

        self.model.fit(
            X_train,
            clicks_train,
            qid=qid_train,
            eval_set=[
                (X_train, clicks_train),
                (X_eval, clicks_eval),
            ],
            eval_qid=[qid_train, qid_eval],
            verbose=10,
        )

    def predict(self, eval_dataset):
        pos_eval = np.ones((len(eval_dataset), 1), dtype=np.float32)
        X_eval = np.concatenate((eval_dataset.features, pos_eval), axis=1)
        return self.model.predict(X_eval)
