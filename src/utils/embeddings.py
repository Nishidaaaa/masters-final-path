from urllib.parse import urlparse
import pickle
import numpy as np

import mlflow


def log_embeddings(model, trainer, datamodule):
    train = datamodule.train_dataloader(shuffle=False, original=True)
    val = datamodule.val_dataloader(shuffle=False, original=True)
    test = datamodule.test_dataloader(shuffle=False, original=True)

    dataloaders = {"val": val, "test": test, "train": train, }
    codes = {}
    labels = {}
    for phase, loader in dataloaders.items():
        embeds_save_path = urlparse(mlflow.get_artifact_uri(f"embeddings/{phase}_data.pickle")).path
        labels_save_path = urlparse(mlflow.get_artifact_uri(f"embeddings/{phase}_label.pickle")).path

        try:
            with open(embeds_save_path, "rb") as f:
                codes[phase] = pickle.load(f)
            with open(labels_save_path, "rb") as f:
                labels[phase] = pickle.load(f)
        except:
            embeds, _labels = [], []
            for embed, label in trainer.predict(model, loader):
                embeds.append(embed.numpy())
                _labels.append(label.numpy())
            embeds = np.concatenate(embeds)
            _labels = np.concatenate(_labels)

            with open(f"tmp/{phase}_data.pickle", "wb") as f:
                pickle.dump(embeds, f, protocol=4)
            with open(f"tmp/{phase}_label.pickle", "wb") as f:
                pickle.dump(_labels, f, protocol=4)

            mlflow.log_artifact(local_path=f"tmp/{phase}_data.pickle", artifact_path=f"embeddings")
            mlflow.log_artifact(local_path=f"tmp/{phase}_label.pickle", artifact_path=f"embeddings")
            codes[phase] = embeds
            labels[phase] = _labels

    return codes, labels
