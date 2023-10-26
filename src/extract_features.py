import pickle
from typing import Tuple, Dict
from urllib.parse import urlparse

import mlflow
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from src import Phase
from src.lightning.datamodule import (PatchDataModule, PatchFrom2ImagesDataModule,
                                      PatchRandomSampleDataModule)
from src.lightning.dino import LDino


def prepare(mode: str, datamodule_args: DictConfig, experiment_name: str, trainer_args: DictConfig) -> Tuple[PatchDataModule, pl.Trainer]:
    """使用するDatamoduleとTrainerを生成して返します．

    Args:
        mode (str): パッチの生成方法．simple:画像をグリッド上切ってそれぞれをパッチに．from2images:simpleと同様だが，訓練時のみ対象学習時に同じラベルの他画像からパッチを2枚とってくる．randomsample:画像をグリッド上ではなく，ランダム座標から切り取る．
        datamodule_args (DictConfig): `PatchDataModule`のコンストラクタ引数
        experiment_name (str):
        trainer_args (DictConfig): `pl.Trainer`のコンストラクタ引数

    Returns:
        Tuple[PatchDataModule, pl.Trainer]: _description_
    """

    data_modules = {"simple": PatchDataModule, "from2images": PatchFrom2ImagesDataModule, "randomsample": PatchRandomSampleDataModule}
    data_module = data_modules[mode](datamodule_args)
    data_module.setup()

    run_id = mlflow.active_run().info.run_id
    logger = MLFlowLogger(experiment_name=experiment_name, run_id=run_id)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", min_delta=0.01, patience=2)
    artifact_path = urlparse(mlflow.get_artifact_uri()).path
    checkpoint = ModelCheckpoint(dirpath=artifact_path, filename="best", auto_insert_metric_name=False, save_last=True, monitor="val_loss")
    trainer = pl.Trainer(logger=logger, callbacks=[lr_monitor, early_stopping, checkpoint], **trainer_args)

    return data_module, trainer


def train_model(trainer: pl.Trainer, datamodule: PatchDataModule, model_args: DictConfig) -> LDino:
    """DINOを訓練して保存し，返します．すでに保存されていた場合は読み込んで返します．

    Args:
        trainer (pl.Trainer): _description_
        datamodule (PatchDataModule): _description_
        model_args (DictConfig): _description_

    Returns:
        LDino: _description_
    """
    try:
        check_point_uri = mlflow.get_artifact_uri("best.ckpt")
        check_point_path = urlparse(check_point_uri).path
        print(f"trying load {check_point_path}...")
        #print("finish 1")
        model = LDino.load_from_checkpoint(check_point_path)
        #print("finish 2")
        #print(" loaded!")
    except:
        model = LDino(model_args)
        #print("finish 3")
        trainer.fit(model, datamodule) # error部分? 
        #print("finish 4")
        trainer.test(model, datamodule)
        #print("finish 5")

    return model


def get_embeddings(data_module: PatchDataModule, model: LDino, trainer: pl.Trainer) -> Dict[Phase, list]:
    """訓練したDINOを用いて，画像の特徴量を抽出して，保存し返します．すでに保存されていた場合は読み込んで返します．

    Args:
        data_module (PatchDataModule): _description_
        model (LDino): _description_
        trainer (pl.Trainer): _description_

    Returns:
        dict: _description_
    """
    train= data_module.train_dataloader(shuffle=False, original=True)
    val= data_module.val_dataloader(shuffle=False, original=True)
    test= data_module.test_dataloader(shuffle=False, original=True)

    dataloaders= {"val": val, "test": test, "train": train}
    codes= {}
    for phase, loader in dataloaders.items():
        embeds_save_path= urlparse(mlflow.get_artifact_uri(f"embeddings/{phase}_data.pickle")).path
        labels_save_path= urlparse(mlflow.get_artifact_uri(f"embeddings/{phase}_label.pickle")).path

        try:
            with open(embeds_save_path, "rb") as f:
                codes[phase]= pickle.load(f)
        except:
            embeds, _labels= [], []
            model.eval()
            for embed, label in trainer.predict(model, loader):
                embeds.append(embed.numpy())
                _labels.append(label.numpy())
                #_labels.append(np.asarray(label))
            embeds= np.concatenate(embeds)
            _labels= np.concatenate(_labels)

            with open(f"tmp/{phase}_data.pickle", "wb") as f:
                pickle.dump(embeds, f, protocol=4)
            with open(f"tmp/{phase}_label.pickle", "wb") as f:
                pickle.dump(_labels, f, protocol=4)

            mlflow.log_artifact(local_path=f"tmp/{phase}_data.pickle", artifact_path=f"embeddings")
            mlflow.log_artifact(local_path=f"tmp/{phase}_label.pickle", artifact_path=f"embeddings")
            codes[phase]= embeds

    return codes


def extract_features(experiment_name: str, mode: str, trainer: DictConfig, model: DictConfig, data_module: DictConfig) -> Tuple[PatchDataModule, LDino, Dict[Phase, list]]:
    """画像から特徴量を抽出するDINOを訓練し、使用したデータモジュール、訓練済みモデル、抽出した特徴量を保存し、返します．モデル，特徴量がすでに保存されていた場合は，訓練を行わず、復元して返します．

    Args:
        experiment_name (str): 現在のmlflow experiment_name
        mode (str): データ取得の方法,simple:画像をグリッド状に区切って，それぞれをパッチとする, from2images:パッチを2つの同ラベル別画像からとってくる, randomsample:パッチをグリッドからではなく，ランダム座標から切り出す．
        trainer (DictConfig): Trainerクラスのコンストラクタ引数
        model (DictConfig): LDinoクラスのコンストラクタ引数
        data_module (DictConfig): PatchDataModuleクラスのコンストラクタ引数

    Returns:
        Tuple[PatchDataModule, LDino, Dict[Phase, list]]: PatchDataModule, LDino, 抽出した特徴量のタプルを返す．
    """
    _data_module, _trainer= prepare(mode=mode, datamodule_args=data_module, experiment_name=experiment_name,  trainer_args=trainer)
    _model= train_model(trainer=_trainer, datamodule=_data_module, model_args=model)
    _embeds= get_embeddings(data_module=_data_module, model=_model, trainer=_trainer)
    return _data_module, _model, _embeds