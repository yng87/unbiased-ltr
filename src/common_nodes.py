import lightning as L
import pandas as pd
import torch
from data_utils import get_data_loader, get_features, get_group, get_unbiased_labels
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from metrics import calc_mean_ndcg
from nn_modules import GradientReversalTwoTower, TwoTower, TwoTowerBase
from sklearn.preprocessing import StandardScaler
from torch import nn

import wandb

N_FEATURES = 136


def train_two_tower(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    batch_size: int,
    max_position: int,
    dropout_prob: float,
    grad_rev_scale: float,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    k: int,
):
    scaler = StandardScaler()

    train_loader = get_data_loader(
        df=df_train,
        scaler=scaler,
        max_position=max_position,
        batch_size=batch_size,
        shuffle=True,
        training=True,
    )
    val_loader = get_data_loader(
        df=df_val,
        scaler=scaler,
        max_position=max_position,
        batch_size=batch_size,
        shuffle=False,
        training=False,
    )

    relevance_tower = nn.Sequential(
        nn.Linear(N_FEATURES, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    observation_tower = nn.Sequential(
        nn.Embedding(num_embeddings=max_position + 1, embedding_dim=64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Dropout(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),
    )
    if grad_rev_scale > 0:
        model: TwoTowerBase = GradientReversalTwoTower(
            relevance_tower=relevance_tower,
            observation_tower=observation_tower,
            k=k,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            grad_rev_scale=grad_rev_scale,
        )
    else:
        model = TwoTower(
            relevance_tower=relevance_tower,
            observation_tower=observation_tower,
            k=k,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    early_stop_callback = EarlyStopping(
        monitor=f"val_ndcg{k}_epoch", patience=5, mode="max"
    )
    checkpoint_callback = ModelCheckpoint(
        # dirpath=checkpoint_dir,
        save_top_k=1,
        monitor=f"val_ndcg{k}_epoch",
        mode="max",
    )
    wandb_logger = WandbLogger()
    trainer = L.Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback],
        logger=wandb_logger,
        accelerator="cpu",
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    return scaler, checkpoint_callback.best_model_path


def predict_by_two_tower(
    scaler: StandardScaler,
    best_model_path: str,
    df: pd.DataFrame,
    grad_rev_scale: float = -1,
) -> pd.DataFrame:
    X = get_features(df).to_numpy()
    X = scaler.transform(X)
    X = torch.tensor(X)
    y = get_unbiased_labels(df)
    group = get_group(df)

    if grad_rev_scale > 0:
        model = GradientReversalTwoTower.load_from_checkpoint(
            best_model_path
        ).relevance_tower
    else:
        model = TwoTower.load_from_checkpoint(best_model_path).relevance_tower
    model = model.to("cpu")
    model.eval()

    preds = model(X).detach().numpy().reshape(-1)

    return pd.DataFrame({"group": group, "label": y, "prediction": preds})


def evaluate(df: pd.DataFrame, k: int) -> dict[str, float]:
    ndcg = calc_mean_ndcg(df, k)
    results = {f"NDCG@{k}": ndcg}
    wandb.log(results)
    return results
