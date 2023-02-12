from urllib.parse import urlparse

import hydra
import mlflow
import numpy as np
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from src.classifier import KmeansGaussian
from src.datamodule import ResizeDataModule
from src.dino import LDino, LViT
from src.preprocess import preprocess
from src.utils.attention import save_attentions
from src.utils.embeddings import log_embeddings
from src.utils.log import log_config, recover_config
import torchmetrics
import torch.nn.functional as F
import torch
experiment_name = "run_resized"


def pred(model, trainer, datamodule):
    train = datamodule.train_dataloader(shuffle=False)
    val = datamodule.val_dataloader(shuffle=False)
    test = datamodule.test_dataloader(shuffle=False)
    dataloaders = {"val": val, "test": test, "train": train, }
    metric = torchmetrics.Accuracy(task="binary", num_classes=2)

    for phase, loader in dataloaders.items():
        preds = []
        _labels = []
        for pred, label in trainer.predict(model, loader):
            preds.append(pred.numpy())
            _labels.append(label.numpy())
        preds = np.concatenate(preds)
        _labels = np.concatenate(_labels)
        acc = metric(torch.Tensor(preds), F.one_hot(torch.Tensor(_labels).to(torch.int64), num_classes=2)).numpy()
        mlflow.log_metric(f"final_{phase}_acc", acc)
        print(phase, acc)


@hydra.main(config_path="./config", config_name=experiment_name, version_base="1.2")
def run_all(config: omegaconf.DictConfig) -> None:
    experiment = mlflow.set_experiment(experiment_name)

    run_id = config.run_main_id
    with mlflow.start_run(run_id=run_id, experiment_id=experiment.experiment_id):
        run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("run_id", run_id)

        current_config = recover_config(config.run_main)
        log_config(current_config)

        logger = MLFlowLogger(experiment_name=experiment_name, run_id=run_id)
        lr_monitor = LearningRateMonitor(logging_interval='step')
        # early_stopping = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001)
        early_stopping = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.0001, patience=2)

        artifact_path = urlparse(mlflow.get_artifact_uri()).path
        checkpoint1 = ModelCheckpoint(dirpath=artifact_path, filename="final")
        checkpoint2 = ModelCheckpoint(dirpath=artifact_path, save_last=True)
        callbacks = [lr_monitor,  checkpoint1, checkpoint2, early_stopping]
        trainer = pl.Trainer(logger=logger, callbacks=callbacks, **current_config.trainer)
        resized_data = ResizeDataModule(current_config.data)
        resized_data.setup()

        try:
            check_point_uri = mlflow.get_artifact_uri("final.ckpt")
            check_point_path = urlparse(check_point_uri).path
            print(f"trying load {check_point_path}...", end="")
            model = LViT.load_from_checkpoint(check_point_path)
            print(" loaded!")
        except:
            print("failed. Start fitting.")
            model = LViT(current_config.model)
            trainer.fit(model, resized_data)
            trainer.test(model, resized_data)
        pred(model, trainer, resized_data)


if __name__ == "__main__":
    run_all()
