from urllib.parse import urlparse

import hydra
import mlflow
import numpy as np
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from src.old_classifier import get_classifier
from src.datamodule import PatchDataModule, PatchFrom2ImagesDataModule, PatchRandomSampleDataModule
from src.dino import LDino
from src.preprocess import reduce_dimension
from src.utils.attention import save_attentions
from src.utils.embeddings import log_embeddings
from src.utils.log import IDLogger, log_config, recover_config
from src.utils.analyze import log_graphs
experiment_name = "run_dino"


def run_dino(config):
    run_id = mlflow.active_run().info.run_id
    logger = MLFlowLogger(experiment_name=experiment_name, run_id=run_id)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.01, patience=2)
    artifact_path = urlparse(mlflow.get_artifact_uri()).path
    checkpoint1 = ModelCheckpoint(dirpath=artifact_path, filename="final")
    checkpoint2 = ModelCheckpoint(dirpath=artifact_path, save_last=True)

    trainer = pl.Trainer(logger=logger, callbacks=[lr_monitor, early_stopping, checkpoint1, checkpoint2], **config.trainer)
    datamodules = {"simple": PatchDataModule, "from2images": PatchFrom2ImagesDataModule, "randomsample": PatchRandomSampleDataModule}

    data = datamodules[config.mode](config.datamodule)
    data.setup()
    try:
        check_point_uri = mlflow.get_artifact_uri("last.ckpt")
        check_point_path = urlparse(check_point_uri).path
        print(f"trying load {check_point_path}...")
        model = LDino.load_from_checkpoint(check_point_path)
        print(" loaded!")
    except:
        print("failed. Start fitting.")
        model = LDino(config.model)
        # trainer.fit(model, data)
        # trainer.test(model, data)
    codes, labels = log_embeddings(model, trainer, data)
    return model, data, codes, labels


@hydra.main(config_path="./config", config_name=experiment_name, version_base="1.2")
def run_all(config: omegaconf.DictConfig) -> None:
    experiment = mlflow.set_experiment(experiment_name)
    id_logger = IDLogger()

    with mlflow.start_run(run_id=config.run_dino_id, experiment_id=experiment.experiment_id):
        id_logger.log("run_dino_id")

        current_config = recover_config(config.run_dino)
        log_config(current_config)

        model, data_module, codes, labels = run_dino(current_config)
        datasets = data_module.patch_datasets

        with mlflow.start_run(run_id=config.preprocess_id, nested=True):
            id_logger.log("preprocess_id")

            current_config = recover_config(config.preprocess)
            log_config(current_config)

            preprocessed = reduce_dimension(current_config, codes)

            with mlflow.start_run(run_id=config.classify_id, nested=True):
                id_logger.log("classify_id")

                current_config = recover_config(config.classify)
                log_config(current_config)
                classifier = get_classifier(current_config.mode, current_config.classifier)
                classifier.fit(datasets, preprocessed)
                preds = classifier.preds
                hists = classifier.hists
                log_graphs(data_module=data_module, preprocessed=preprocessed, preds=preds, clustered=classifier.clustered, hists=hists,
                           n_clusters=config.classify.classifier.kmeans_args.n_clusters, image_size=config.common.image_size, probas=classifier.probas)

                # try:
                #     save_attentions(model, current_config.classifier.kmeans_args.n_clusters,
                #                     np.array(classifier.clustered["train"]), datasets["train"])
                # except:
                #     print("no attentions provided.")


if __name__ == "__main__":
    run_all()
