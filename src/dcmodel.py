import pytorch_lightning as pl
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import src
from src.utils.data import LabellablePatchDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import torch


class LViTCluster(pl.LightningModule):
    def __init__(self, vit, hparams, datasets, init_labels) -> None:
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        self.original_datasets = datasets
        self.datasets = {
            phase: LabellablePatchDataset(self.original_datasets[phase]) for phase in src.PHASES
        }
        for phase in src.PHASES:
            self.datasets[phase].set_labels(np.array(init_labels[phase], dtype=np.float32)/16)

        self.vit = vit
        self.head = nn.LazyLinear(1)
        self.sigmoid = nn.Sigmoid()

        self.embeds = []

    def _hook(self, _, input, output):
        self.embeds.extend(output.clone().detach().tolist())

    def forward(self, x):
        x = self.vit(x)
        x = self.head(x)
        x = self.sigmoid(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        scheduler = CosineAnnealingLR(optimizer, 111)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def step(self, batch):

        loss = nn.MSELoss()
        x, y = batch
        y = y.to(torch.float32).unsqueeze(-1)
        pred = self(x)
        return loss(pred, y), pred, y

    def training_step(self, batch, batch_idx):
        loss, pred, y = self.step(batch)
        logs = {"loss": loss}
        self.log_dict(logs)
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_idx):
        loss, pred, y = self.step(batch)
        logs = {"val_loss": loss}
        self.log_dict(logs)
        return {"val_loss": loss, "log": logs}

    def test_step(self, batch, batch_idx):
        loss, pred, y = self.step(batch)
        logs = {"test_loss": loss}
        self.log_dict(logs)
        return {"test_loss": loss, "log": logs}

    def get_loader(self, phase):
        return DataLoader(self.datasets[phase], batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers)

    def train_dataloader(self,shuffle=False):
        return self.get_loader("train")

    def val_dataloader(self):
        return self.get_loader("val")

    def test_dataloader(self):
        return self.get_loader("test")

    def on_train_epoch_start(self) -> None:
        self.embeds = []
        return super().on_train_epoch_start()

    def on_train_batch_start(self, batch, batch_idx: int):
        self.handle = self.vit.register_forward_hook(self._hook)
        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx: int) -> None:
        self.handle.remove()
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self) -> None:
        labels = self.datasets["train"].dataset.labels
        pca = PCA(n_components=4)
        codes = pca.fit_transform(self.embeds)
        kmeans = KMeans(16)
        clusters = np.array(kmeans.fit_predict(codes))

        def sortclusters(c, l):
            ca = [0]*16
            ng = [0]*16
            for i, x in enumerate(c):
                if l[i]:
                    ca[x] += 1
                else:
                    ng[x] += 1
            ca = np.array(ca)
            ng = np.array(ng)
            ratio = ca/(ng+ca)
            new_map = np.argsort(ratio)
            new_cluster = []
            for x in c:
                new_cluster.append(np.argmax(new_map == x))
            return new_cluster

        clusters = np.array(sortclusters(clusters, labels), dtype=np.float32)
        clusters /= 16.0
        self.datasets["train"].set_labels(clusters)

        return super().on_train_epoch_end()
