import numpy as np
import pandas as pd
from tqdm import tqdm

NUM_FEATURES = 136
MAX_LABEL = 4


def preprocess_raw_web30k_dataset(raw_data: str) -> pd.DataFrame:
    lines = raw_data.strip().split("\n")
    length = len(lines)

    labels = np.empty(shape=(length,), dtype=np.int32)
    queries = np.empty(shape=(length,), dtype=np.int32)
    features = np.empty(shape=(length, NUM_FEATURES), dtype=np.float32)

    for i, line in tqdm(enumerate(lines)):
        elms = line.strip().split(" ")
        labels[i] = elms[0]
        queries[i] = elms[1].split(":")[1]
        feature = np.array(
            [float(x.split(":")[1]) for x in elms[2:]],
            dtype=np.float32,
        )
        features[i] = feature

    return pd.DataFrame(
        {
            "label": labels,
            "query": queries,
            **{f"feature_{i}": features[:, i] for i in range(NUM_FEATURES)},
        }
    )


def _generate_positions(
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
        positions[idx_from:idx_to] = np.argsort(-scores[idx_from:idx_to]) + 1
    return positions


def _generate_observation_probs(positions: np.ndarray) -> np.ndarray:
    return 1 / positions


def _generate_clicks(
    labels: np.ndarray,
    click_noise: float,
    observation_probs: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    click_probs = click_noise + (1 - click_noise) * (2**labels - 1) / (2**MAX_LABEL - 1)
    clicks = rng.binomial(1, observation_probs * click_probs)
    return clicks.astype(np.int32)


def generate_synthetic_clicks_dataset(
    web30k_dataset: pd.DataFrame,
    click_noise: float,
    oracle_weight: float,
    random_seed: int,
) -> pd.DataFrame:
    if click_noise < 0 or click_noise > 1:
        raise ValueError("click_noise must be between 0 and 1")

    rng = np.random.default_rng(random_seed)

    df = web30k_dataset.sort_values(by="query")

    labels = df["label"].to_numpy()
    queries = df["query"].to_numpy()

    group_boundary_indices = (
        np.concatenate(([True], queries[1:] != queries[:-1], [True]))
    ).nonzero()[0]

    positions = _generate_positions(labels, group_boundary_indices, oracle_weight, rng)
    observation_probs = _generate_observation_probs(positions)
    clicks = _generate_clicks(labels, click_noise, observation_probs, rng)

    df["click"] = clicks
    df["position"] = positions

    return df


def negative_down_sample(
    df: pd.DataFrame, ratio: float, random_seed: int
) -> pd.DataFrame:
    if ratio < 0 or ratio > 1:
        raise ValueError("ratio must be between 0 and 1")

    rng = np.random.default_rng(random_seed)

    pos_mask = df["click"] == 1
    neg_mask = (df["click"] == 0) & (rng.uniform(size=len(df)) < ratio)
    mask = pos_mask | neg_mask

    return df[mask]
