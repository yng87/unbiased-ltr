import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[df.columns[df.columns.str.startswith("feature")]]


def get_position(df: pd.DataFrame, max_position: int = 0) -> pd.Series:
    pos = df["position"]
    if max_position > 0:
        pos[pos >= max_position] = max_position
    return pos


def get_unbiased_labels(df: pd.DataFrame) -> pd.Series:
    return df["label"]


def get_clicks(df: pd.DataFrame) -> pd.Series:
    return df["click"]


def get_group(df: pd.DataFrame) -> pd.Series:
    return df["query"]


def sort_by_group(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by="query")


class Web30KSyntheticClickDataset(Dataset):
    def __init__(self, features, clicks, groups, positions=None):
        self.features = features
        self.clicks = clicks
        self.groups = groups
        self.positions = positions

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        data = {
            "features": self.features[idx],
            "clicks": self.clicks[idx],
            "groups": self.groups[idx],
        }
        if self.positions is not None:
            data["positions"] = self.positions[idx]
        return data


def get_data_loader(
    df: pd.DataFrame,
    scaler: StandardScaler,
    max_position: int,
    batch_size: int,
    shuffle: bool,
    training: bool,
) -> DataLoader:
    X = get_features(df).to_numpy(dtype=np.float32)
    if max_position > 0:
        positions = get_position(df, max_position=max_position).to_numpy(dtype=np.int32)
    else:
        positions = None
    y = get_clicks(df).to_numpy(dtype=np.float32)
    group = get_group(df).to_numpy(dtype=np.int32)

    if training:
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    dataset = Web30KSyntheticClickDataset(X, y, group, positions)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return data_loader
