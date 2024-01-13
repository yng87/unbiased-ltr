import json
import os

import numpy as np

from datasets import WEB30KDataset, WEB30KSyntheticClickDataset
from metrics import calc_mean_ndcg
from models.base import ModelBase


class Evaluator:
    DATASET_DIR = "datasets/MSLR-WEB30K-Fold1/processed/"
    K = 5
    ORACLE_WEIGHTS = [0, 0.2, 0.6, 0.8, 1]

    def __init__(self, negative_down_sample_ratio: float, save_path: str):
        self._rng = np.random.default_rng(42)
        self._negative_down_sample_ratio = negative_down_sample_ratio
        self._save_path = save_path

    def save_result(self, results):
        os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
        with open(self._save_path, "w") as f:
            json.dump(results, f, indent=1)

    def evaluate(self, model: ModelBase):
        results = {}
        for oracle_weight in self.ORACLE_WEIGHTS:
            print("-" * 40)
            train_dataset = WEB30KSyntheticClickDataset.load(
                os.path.join(
                    self.DATASET_DIR,
                    f"synthetic_train_{int(oracle_weight*100)}.npz",
                )
            ).negative_down_sample(self._negative_down_sample_ratio, self._rng)
            valid_dataset = WEB30KSyntheticClickDataset.load(
                os.path.join(
                    self.DATASET_DIR,
                    f"synthetic_valid_{int(oracle_weight*100)}.npz",
                )
            )
            test_dataset = WEB30KDataset.load(
                os.path.join(self.DATASET_DIR, "test.npz")
            )

            model.build()
            model.fit(train_dataset=train_dataset, eval_dataset=valid_dataset)
            preds = model.predict(test_dataset)

            ndcg = calc_mean_ndcg(
                test_dataset.group_boundary_indices,
                labels=test_dataset.labels,
                preds=preds,
                k=self.K,
            )
            print(f"oracle_weight={oracle_weight}, NDCG@{self.K}={ndcg}")
            results[oracle_weight] = {f"NDCG@{self.K}": ndcg}

        self.save_result(results)
