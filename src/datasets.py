import numpy as np
from tqdm import tqdm

LEN_TRAIN = 2270296
LEN_VALID = 747218
LEN_TEST = 753611
NUM_FEATURES = 136
MAX_LABEL = 4
CLICK_NOISE = 0.1


class WEB30KDataset:
    def __init__(self, path: str, length: int):
        self.labels = np.empty(shape=(length,), dtype=np.int32)
        self.queries = np.empty(shape=(length,), dtype=np.int32)
        self.features = np.empty(
            shape=(length, NUM_FEATURES), dtype=np.float32
        )

        with open(path, "r") as f:
            for i, line in tqdm(enumerate(f)):
                elms = line.strip().split(" ")
                self.labels[i] = elms[0]
                self.queries[i] = elms[1].split(":")[1]
                feature = np.array(
                    [float(x.split(":")[1]) for x in elms[2:]],
                    dtype=np.float32,
                )
                self.features[i] = feature

        sorted_idx = np.argsort(self.queries)
        self.labels = self.labels[sorted_idx]
        self.queries = self.queries[sorted_idx]
        self.features = self.features[sorted_idx]

        self.group_boundary_indices = (
            np.concatenate(
                ([True], self.queries[1:] != self.queries[:-1], [True])
            )
        ).nonzero()[0]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "queries": self.queries[idx],
            "features": self.features[idx],
            "labels": self.labels[idx],
        }

    def save(self, path: str):
        np.savez_compressed(
            path,
            queries=self.queries,
            features=self.features,
            labels=self.labels,
            group_boundary_indices=self.group_boundary_indices,
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        dataset = WEB30KDataset.__new__(WEB30KDataset)
        dataset.queries = data["queries"]
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.group_boundary_indices = data["group_boundary_indices"]
        return dataset


class WEB30KSyntheticClickDataset:
    def __init__(
        self,
        dataset: WEB30KDataset,
        rng: np.random.Generator,
        oracle_weight: float,
    ):
        self.queries = dataset.queries
        self.features = dataset.features
        self.labels = dataset.labels
        self.group_boundary_indices = dataset.group_boundary_indices

        self.positions = self._generate_positions(
            labels=self.labels,
            group_boundary_indices=self.group_boundary_indices,
            weight=oracle_weight,
            rng=rng,
        )
        self.observation_probs = self._generate_observation_probs(
            positions=self.positions
        )
        self.clicks = self._generate_clicks(
            labels=self.labels,
            observation_probs=self.observation_probs,
            rng=rng,
        )

    def _generate_positions(
        self,
        labels: np.ndarray,
        group_boundary_indices: np.ndarray,
        weight: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        noises = rng.uniform(low=0, high=MAX_LABEL, size=len(labels))
        scores = weight * labels + (1 - weight) * noises
        positions = np.empty(shape=(len(labels),), dtype=np.int32)
        for i in range(len(group_boundary_indices) - 1):
            idx_from = group_boundary_indices[i]
            idx_to = group_boundary_indices[i + 1]
            positions[idx_from:idx_to] = (
                np.argsort(-scores[idx_from:idx_to]) + 1
            )
        return positions

    def _generate_observation_probs(self, positions: np.ndarray) -> np.ndarray:
        return 1 / positions

    def _generate_clicks(
        self,
        labels: np.ndarray,
        observation_probs: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        click_probs = CLICK_NOISE + (1 - CLICK_NOISE) * (2**labels - 1) / (
            2**MAX_LABEL - 1
        )
        clicks = rng.binomial(1, observation_probs * click_probs)
        return clicks.astype(np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "queries": self.queries[idx],
            "features": self.features[idx],
            "clicks": self.clicks[idx],
            "positions": self.positions[idx],
        }

    def negative_down_sample(self, ratio: float, rng: np.random.Generator):
        pos_mask = self.clicks == 1
        neg_mask = (self.clicks == 0) & (rng.uniform(size=len(self)) < ratio)
        mask = pos_mask | neg_mask

        dataset = WEB30KSyntheticClickDataset.__new__(
            WEB30KSyntheticClickDataset
        )
        dataset.queries = self.queries[mask]
        dataset.features = self.features[mask]
        dataset.labels = self.labels[mask]
        dataset.positions = self.positions[mask]
        dataset.observation_probs = self.observation_probs[mask]
        dataset.clicks = self.clicks[mask]
        dataset.group_boundary_indices = (
            np.concatenate(
                ([True], dataset.queries[1:] != dataset.queries[:-1], [True])
            )
        ).nonzero()[0]
        return dataset

    def save(self, path: str):
        np.savez_compressed(
            path,
            queries=self.queries,
            features=self.features,
            labels=self.labels,
            positions=self.positions,
            observation_probs=self.observation_probs,
            group_boundary_indices=self.group_boundary_indices,
            clicks=self.clicks,
        )

    @classmethod
    def load(cls, path: str):
        data = np.load(path)
        dataset = WEB30KSyntheticClickDataset.__new__(
            WEB30KSyntheticClickDataset
        )
        dataset.queries = data["queries"]
        dataset.features = data["features"]
        dataset.labels = data["labels"]
        dataset.positions = data["positions"]
        dataset.observation_probs = data["observation_probs"]
        dataset.group_boundary_indices = data["group_boundary_indices"]
        dataset.clicks = data["clicks"]
        return dataset


def transform_and_save_datasets():
    import os

    dataset_dir = "datasets/MSLR-WEB30K-Fold1/"
    train_dataset = WEB30KDataset(
        os.path.join(dataset_dir, "train.txt"), LEN_TRAIN
    )
    valid_dataset = WEB30KDataset(
        os.path.join(dataset_dir, "vali.txt"), LEN_VALID
    )
    test_dataset = WEB30KDataset(
        os.path.join(dataset_dir, "test.txt"), LEN_TEST
    )

    save_dir = os.path.join(dataset_dir, "processed")
    os.makedirs(save_dir, exist_ok=True)

    test_dataset.save(os.path.join(save_dir, "test.npz"))

    rng = np.random.default_rng(42)
    oracle_weights = [0.0, 0.2, 0.6, 0.8, 1.0]
    for oracle_weight in tqdm(oracle_weights):
        synthetic_train_dataset = WEB30KSyntheticClickDataset(
            train_dataset, rng, oracle_weight=oracle_weight
        )
        synthetic_valid_dataset = WEB30KSyntheticClickDataset(
            valid_dataset, rng, oracle_weight=oracle_weight
        )
        synthetic_test_dataset = WEB30KSyntheticClickDataset(
            test_dataset, rng, oracle_weight=oracle_weight
        )

        synthetic_train_dataset.save(
            os.path.join(
                save_dir,
                f"synthetic_train_{int(oracle_weight * 100)}.npz",
            )
        )
        synthetic_valid_dataset.save(
            os.path.join(
                save_dir,
                f"synthetic_valid_{int(oracle_weight * 100)}.npz",
            )
        )
        synthetic_test_dataset.save(
            os.path.join(
                save_dir,
                f"synthetic_test_{int(oracle_weight * 100)}.npz",
            )
        )


if __name__ == "__main__":
    transform_and_save_datasets()

    # check if the logged dataset is correctly saved
    logged_train_dataset = WEB30KSyntheticClickDataset.load(
        "datasets/MSLR-WEB30K-Fold1/processed/synthetic_train_0.npz"
    )
    print(len(logged_train_dataset))
    print(logged_train_dataset.clicks[:100])
