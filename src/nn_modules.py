import lightning as L
import torch
from metrics import NDCG
from torch import nn, optim
from torch.autograd import Function


class SingleTower(L.LightningModule):
    N_FEATURES = 136

    def __init__(self, learning_rate: float, weight_decay: float, k: int):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_ndcg = NDCG(k=k)
        self.val_ndcg = NDCG(k=k)
        self.model = nn.Sequential(
            nn.Linear(self.N_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        clicks = batch["clicks"].reshape(-1, 1).float()
        groups = batch["groups"]

        logits = self.model(features)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, clicks)
        self.log("train_loss", loss)

        self.train_ndcg(logits, clicks, groups)

        return loss

    def on_train_epoch_end(self):
        self.log(
            f"train_ndcg{self.train_ndcg.k}_epoch",
            self.train_ndcg,
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        clicks = batch["clicks"].reshape(-1, 1).float()
        groups = batch["groups"]

        logits = self.model(features)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, clicks)
        self.log("val_loss", loss)

        self.val_ndcg(logits, clicks, groups)

        return loss

    def on_validation_epoch_end(self):
        self.log(
            f"val_ndcg{self.val_ndcg.k}_epoch",
            self.val_ndcg,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


class TwoTowerBase(L.LightningModule):
    N_FEATURES = 136

    def __init__(
        self,
        relevance_tower: nn.Module,
        observation_tower: nn.Module,
        learning_rate: float,
        weight_decay: float,
        k: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.relevance_tower = relevance_tower
        self.observation_tower = observation_tower
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_ndcg = NDCG(k=k)
        self.val_ndcg = NDCG(k=k)

    def _calc_loss(self, features, clicks, positions):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        clicks = batch["clicks"].reshape(-1, 1).float()
        positions = batch["positions"]
        groups = batch["groups"]

        logits, loss = self._calc_loss(
            features=features,
            clicks=clicks,
            positions=positions,
        )
        self.log("train_loss", loss)

        self.train_ndcg(logits, clicks, groups)

        return loss

    def on_train_epoch_end(self):
        self.log(
            f"train_ndcg{self.train_ndcg.k}_epoch",
            self.train_ndcg,
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        clicks = batch["clicks"].reshape(-1, 1).float()
        positions = batch["positions"]
        groups = batch["groups"]

        _, loss = self._calc_loss(
            features=features,
            clicks=clicks,
            positions=positions,
        )
        self.log("val_loss", loss)

        preds = self.relevance_tower(features)
        self.val_ndcg(preds, clicks, groups)

        return loss

    def on_validation_epoch_end(self):
        self.log(
            f"val_ndcg{self.val_ndcg.k}_epoch",
            self.val_ndcg,
            on_step=False,
            on_epoch=True,
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer


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


class TwoTower(TwoTowerBase):
    def _calc_loss(self, features, clicks, positions):
        loss_fn = nn.BCEWithLogitsLoss()
        r = self.relevance_tower(features)
        o = self.observation_tower(positions)
        logits = r + o
        loss = loss_fn(logits, clicks)
        return logits, loss


class GradientReversalTwoTower(TwoTowerBase):
    def __init__(self, grad_rev_scale: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_rev = RevGrad(alpha=grad_rev_scale)
        self.adv_linear = nn.Linear(1, 1)

    def _calc_loss(self, features, clicks, positions):
        loss_fn = nn.BCEWithLogitsLoss()
        adv_loss_fn = nn.MSELoss()
        r = self.relevance_tower(features)
        o = self.observation_tower(positions)
        grad_rev = self.adv_linear(self.grad_rev(o))
        logits = r + o
        loss = loss_fn(logits, clicks) + adv_loss_fn(clicks, grad_rev)
        return logits, loss
