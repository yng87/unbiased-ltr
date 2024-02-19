import logging

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.preprocessing import StandardScaler

from data_utils import get_data_loader, get_features, get_group, get_unbiased_labels
from nn_modules import SingleTower

logger = logging.getLogger(__name__)


def train(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    k: int,
):
    scaler = StandardScaler()
    train_loader = get_data_loader(
        df=df_train,
        scaler=scaler,
        max_position=0,
        batch_size=batch_size,
        shuffle=True,
        training=True,
    )
    val_loader = get_data_loader(
        df=df_val,
        scaler=scaler,
        max_position=0,
        batch_size=batch_size,
        shuffle=False,
        training=False,
    )

    single_tower = SingleTower(
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
        model=single_tower,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )
    return scaler, checkpoint_callback.best_model_path


def predict(
    scaler: StandardScaler,
    best_model_path: str,
    df: pd.DataFrame,
) -> pd.DataFrame:
    X = get_features(df).to_numpy()
    X = scaler.transform(X)
    X = torch.tensor(X)
    y = get_unbiased_labels(df)
    group = get_group(df)

    model = SingleTower.load_from_checkpoint(best_model_path).model
    model = model.to("cpu")
    model.eval()

    preds = model(X).detach().numpy().reshape(-1)

    return pd.DataFrame({"group": group, "label": y, "prediction": preds})
