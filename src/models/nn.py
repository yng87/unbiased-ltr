import copy
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.autograd import Function
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import calc_mean_ndcg
from models.base import ModelBase


class SingleTower(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        features, _ = x
        return self.mlp(features)

    def calc_loss(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss

    def predict(self, features):
        return self.mlp(features)


class TwoTower(nn.Module):
    def __init__(self, n_features: int, max_position: int):
        super().__init__()
        self.max_position = max_position
        self.relevance_tower = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.observation_tower = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.max_position + 1, embedding_dim=64
            ),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        features, positions = x
        positions[positions >= self.max_position] = self.max_position

        r = self.relevance_tower(features)
        o = self.observation_tower(positions)
        return r + o

    def calc_loss(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss

    def predict(self, features):
        return self.relevance_tower(features)


class PAL(nn.Module):
    def __init__(self, n_features: int, max_position: int):
        super().__init__()
        self.max_position = max_position
        self.relevance_tower = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.observation_tower = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.max_position + 1, embedding_dim=64
            ),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        features, positions = x
        positions[positions >= self.max_position] = self.max_position

        r = self.relevance_tower(features)
        o = self.observation_tower(positions)
        return r * o

    def calc_loss(self, probs, labels):
        loss = self.loss_fn(probs, labels)
        return loss

    def predict(self, features):
        return self.relevance_tower(features)


class ObsevationDropoutTwoTower(nn.Module):
    def __init__(
        self, n_features: int, max_position: int, dropout_prob: float
    ):
        super().__init__()
        self.max_position = max_position
        self.relevance_tower = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.observation_tower = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.max_position + 1, embedding_dim=64
            ),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_prob),
            nn.Linear(32, 1),
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        features, positions = x
        positions[positions >= self.max_position] = self.max_position

        r = self.relevance_tower(features)
        o = self.observation_tower(positions)
        return r + o

    def calc_loss(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss

    def predict(self, features):
        return self.relevance_tower(features)


# https://github.com/janfreyberg/pytorch-revgrad
class RevGradFunction(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output * alpha_
        return grad_input, None


revgrad = RevGradFunction.apply


class RevGrad(nn.Module):
    def __init__(self, alpha=1.0, *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, input_):
        return revgrad(input_, self._alpha)


class GradientReversalTwoTower(nn.Module):
    def __init__(
        self, n_features: int, max_position: int, grad_rev_scale: float
    ):
        super().__init__()
        self.max_position = max_position
        self.relevance_tower = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.observation_tower = nn.Sequential(
            nn.Embedding(
                num_embeddings=self.max_position + 1, embedding_dim=64
            ),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.grad_rev = RevGrad(alpha=grad_rev_scale)
        self.adv_linear = nn.Linear(1, 1)

        self.loss_fn = nn.BCEWithLogitsLoss()
        self.adv_loss_fn = nn.MSELoss()

    def forward(self, x):
        features, positions = x
        positions[positions >= self.max_position] = self.max_position

        r = self.relevance_tower(features)
        o = self.observation_tower(positions)
        grad_rev = self.adv_linear(self.grad_rev(o))
        return r + o, grad_rev

    def calc_loss(self, outputs, clicks):
        pred, grad_rev = outputs
        click_loss = self.loss_fn(pred, clicks)
        adv_loss = self.adv_loss_fn(clicks, grad_rev)
        loss = click_loss + adv_loss
        return loss

    def predict(self, features):
        return self.relevance_tower(features)


class NNModel(ModelBase):
    MODEL_MAPPER = {
        "single_tower": SingleTower,
        "two_tower": TwoTower,
        "pal": PAL,
        "observation_dropout": ObsevationDropoutTwoTower,
        "gradient_reversal": GradientReversalTwoTower,
    }

    def __init__(
        self,
        model_name: str,
        model_params: dict[str, Any],
        epochs: int,
        weight_decay: float,
        batch_size=64,
        learning_rate=0.01,
    ):
        torch.manual_seed(42)
        self.model_name = model_name
        self.model_params = model_params
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def build(self):
        self.scaler = StandardScaler()
        self.model = self.MODEL_MAPPER[self.model_name](**self.model_params)

    @staticmethod
    def calc_training_metrics(
        current_samples, current_loss, total_samples, total_loss
    ):
        total_samples += current_samples
        total_loss += (
            current_loss / total_samples
            - current_samples * total_loss / total_samples
        )
        return total_samples, total_loss

    def train_loop(self, dataloader, model, optimizer, epoch) -> float:
        model.train()
        total_loss = 0
        total_samples = 0
        with tqdm(dataloader, desc=f"Epoch {epoch}: Train") as t:
            for batch in t:
                clicks = batch["clicks"].reshape(-1, 1).float()
                features = batch["features"]
                positions = batch["positions"]

                optimizer.zero_grad()

                pred = model((features, positions))
                loss = model.calc_loss(pred, clicks)

                loss.backward()
                optimizer.step()

                total_samples, total_loss = self.calc_training_metrics(
                    current_samples=len(features),
                    current_loss=loss.item(),
                    total_samples=total_samples,
                    total_loss=total_loss,
                )
                t.set_postfix(samples=total_samples, loss=total_loss)

        return total_loss

    def batch_predict(self, dataloader, model, desc):
        model.eval()
        preds = []
        total_samples = 0
        with torch.no_grad():
            with tqdm(dataloader, desc=desc) as t:
                for batch in t:
                    features = batch["features"]
                    pred = self.model.predict(features).reshape(-1)
                    preds.extend(pred.detach().numpy())
                    total_samples += len(features)
                    t.set_postfix(samples=total_samples)

        preds = np.array(preds, dtype=np.float32)
        return preds

    def test_loop(self, dataloader, model, epoch):
        preds = self.batch_predict(
            dataloader=dataloader, model=model, desc=f"Epoch {epoch}: Valid"
        )

        ndcg_click = calc_mean_ndcg(
            group_boundary_indices=dataloader.dataset.group_boundary_indices,
            labels=dataloader.dataset.clicks,
            preds=preds,
            k=5,
        )

        return float(ndcg_click)

    def fit(self, train_dataset, eval_dataset):
        train_dataset.features = self.scaler.fit_transform(
            train_dataset.features
        )
        eval_dataset.features = self.scaler.transform(eval_dataset.features)

        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, shuffle=False
        )
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        patience = 5
        best_metric = 0
        unimproved_count = 0
        best_model = copy.deepcopy(self.model)
        best_epoch = 0
        for t in range(self.epochs):
            self.train_loop(train_dataloader, self.model, optimizer, t + 1)
            val_metric = self.test_loop(eval_dataloader, self.model, t + 1)
            print(
                f"\tVal NDCG@5: current = {val_metric}, best = {best_metric}"
            )
            # early stopping
            if val_metric > best_metric:
                best_metric = val_metric
                unimproved_count = 0
                best_model = copy.deepcopy(self.model)
                best_epoch = t + 1
            else:
                unimproved_count += 1
                if unimproved_count >= patience:
                    print(
                        "\tEearly stopping."
                        f"Get the best model at epoch {best_epoch}"
                    )
                    break

        self.model = best_model

    def predict(self, eval_dataset):
        eval_dataset.features = self.scaler.transform(eval_dataset.features)
        dataloader = DataLoader(
            eval_dataset, batch_size=self.batch_size, shuffle=False
        )
        preds = self.batch_predict(
            dataloader=dataloader, model=self.model, desc="Pred"
        )

        return preds
