

import math
from functools import partial

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import vit_pytorch.dino as dino
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.utils.attention import AttentionGetter


class LDino(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        self.model = timm.create_model(**self.hparams.net)
        self.learner = dino.Dino(self.model, image_size=self.hparams.image_size, **self.hparams.dino)

    def forward(self, x, return_attention=False):
        if return_attention:
            return self.learner(x, return_embedding=True, return_projection=False)
        x = self.learner(x)
        return x

    def configure_optimizers(self):
        optimizer = Adam(self.learner.parameters(), lr=self.hparams.lr, betas=(self.hparams.beta1, self.hparams.beta2))
        scheduler = ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def training_step(self, batch, batch_idx):
        x, _ = batch
        loss = self(x)
        logs = {"loss": loss}
        self.log_dict(logs, on_step=True)
        return {"loss": loss, "log": logs}

    def training_epoch_end(self, outputs):
        self.learner.update_moving_average()
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        logs = {"avg_loss": avg_loss}
        self.log_dict(logs, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        loss = self(x)
        logs = {"val_loss": loss}
        self.log_dict(logs, on_step=True)
        return {"val_loss": loss, "log": logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_loss}
        self.log_dict(logs, on_epoch=True)
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        x, _ = batch
        loss = self(x)

        logs = {"test_loss": loss}
        self.log_dict(logs, on_step=True)
        return {"test_loss": loss, "log": logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"avg_test_loss": avg_loss}
        self.log_dict(logs, on_epoch=True)
        return {"avg_test_loss": avg_loss, "log": logs}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        emb, _ = self.learner.forward(x, return_embedding=True, return_projection=True)
        # emb = self.learner.forward(x, return_embedding=True, return_projection=False)
        return emb, y

    def get_attention_maps(self, input_tensor, expand=False):
        batched_input = input_tensor.unsqueeze(0)
        attention_getter = AttentionGetter(self.model)
        _, attn = attention_getter(batched_input)
        attention_getter.eject()

        bs = attn.shape[0]
        nh = attn.shape[2]
        w = h = int(math.sqrt(attn.shape[-1]-1))
        attn = attn[:, 0, :, 0, 1:].reshape(bs, nh, -1)
        attn = attn.reshape(bs, nh, w, h)
        if expand:
            attn = nn.functional.interpolate(attn, scale_factor=self.hparams.image_size//w, mode="nearest").cpu().numpy()
        attn = attn.squeeze()
        return attn


class From2ImagesDino(dino.Dino):
    def __init__(
        self,
        net,
        image_size,
        hidden_layer=-2,
        projection_hidden_size=256,
        num_classes_K=65336,
        projection_layers=4,
        student_temp=0.9,
        teacher_temp=0.04,
        local_upper_crop_scale=0.4,
        global_lower_crop_scale=0.5,
        moving_average_decay=0.9,
        center_moving_average_decay=0.9,
        augment_fn=None,
        augment_fn2=None
    ):
        try:
            super().__init__(net, image_size, hidden_layer, projection_hidden_size, num_classes_K,
                             projection_layers, student_temp, teacher_temp, local_upper_crop_scale, global_lower_crop_scale, moving_average_decay,
                             center_moving_average_decay, augment_fn, augment_fn2)
        except:
            self.forward(torch.randn(2, 2, 3, image_size, image_size, device=dino.get_module_device(self.net)))

    def forward(
        self,
        x,
        return_embedding=False,
        return_projection=True,
        student_temp=None,
        teacher_temp=None
    ):
        x1, x2 = x
        if return_embedding:
            return self.student_encoder(x1, return_projection=return_projection)

        image_one, image_two = self.augment1(x1), self.augment2(x2)

        local_image_one, local_image_two = self.local_crop(image_one),  self.local_crop(image_two)
        global_image_one, global_image_two = self.global_crop(image_one), self.global_crop(image_two)

        student_proj_one, _ = self.student_encoder(local_image_one)
        student_proj_two, _ = self.student_encoder(local_image_two)

        with torch.no_grad():
            teacher_encoder = self._get_teacher_encoder()
            teacher_proj_one, _ = teacher_encoder(global_image_one)
            teacher_proj_two, _ = teacher_encoder(global_image_two)

        loss_fn_ = partial(
            dino.loss_fn,
            student_temp=dino.default(student_temp, self.student_temp),
            teacher_temp=dino.default(teacher_temp, self.teacher_temp),
            centers=self.teacher_centers
        )

        teacher_logits_avg = torch.cat((teacher_proj_one, teacher_proj_two)).mean(dim=0)
        self.last_teacher_centers.copy_(teacher_logits_avg)

        loss = (loss_fn_(teacher_proj_one, student_proj_two) + loss_fn_(teacher_proj_two, student_proj_one)) / 2
        return loss


class LPatchFrom2ImageDino(LDino):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.save_hyperparameters(hparams)
        self.model = timm.create_model(**self.hparams.net)
        self.learner = From2ImagesDino(self.model, image_size=self.hparams.image_size, **self.hparams.dino)


class LViT(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams, logger=False)
        self.model = timm.create_model(**self.hparams.net)
        self.sm = nn.Softmax(dim=1)
        self.accuracy = torchmetrics.Accuracy(task="binary", num_classes=2)

    def forward(self, x):
        x = self.model(x)
        return F.softmax(x)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), **self.hparams.optimizer)
        scheduler = ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def step(self, batch):
        x, y = batch
        pred = self(x)
        loss = F.cross_entropy(pred, y)
        return pred, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred, loss = self.step(batch)

        log = {"loss": loss, "train_acc": self.accuracy(pred, F.one_hot(y, num_classes=2))}
        self.log_dict(log, on_step=True)

        return {"loss": loss, "log": log}

    def validation_step(self, batch, batch_idx):
        x, y = batch

        pred, loss = self.step(batch)

        log = {"val_loss": loss, "val_acc": self.accuracy(pred, F.one_hot(y, num_classes=2))}
        self.log_dict(log, on_step=True)
        return {"val_loss": loss, "log": log}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"avg_val_loss": avg_loss}
        self.log_dict(logs, on_epoch=True)
        return {"avg_val_loss": avg_loss, "log": logs}

    def test_step(self, batch, batch_idx):
        x, y = batch

        pred, loss = self.step(batch)
        log = {"test_loss": loss, "test_acc": self.accuracy(pred, F.one_hot(y, num_classes=2))}
        self.log_dict(log, on_step=True)
        return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        logs = {"avg_test_loss": avg_loss}
        self.log_dict(logs, on_epoch=True)
        return {"avg_test_loss": avg_loss, "log": logs}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        x, y = batch
        return self(x), y
