from src.utils.ml import PcaUmapReducer, UmapReducer, PcaReducer
from urllib.parse import urlparse
import mlflow
import pickle
import src
from typing import Dict


def get_reducer(mode, features, reducer):
    methods = {"pca": PcaReducer, "pca_umap": PcaUmapReducer, "umap": UmapReducer}
    method = methods[mode]

    _reducer = method.try_load()
    if _reducer is None:
        _reducer = method.from_config(reducer)
        _reducer.train(features["train"])
    return _reducer


def reduce_dimension(features, mode, reducer) -> Dict[src.Phase, list]:
    reducer = get_reducer(mode, features, reducer)
    reduced_features = {}
    for phase in src.PHASES:
        save_path = urlparse(mlflow.get_artifact_uri(f"{phase}.pickle")).path
        try:
            print(f"trying load {save_path}...", end="")
            with open(save_path, "rb") as f:
                reduced_features[phase] = pickle.load(f)
            print("done")
        except:
            print("failed. Start transforming.")
            reduced_features[phase] = reducer.transform(features[phase])
            with open(save_path, "wb") as f:
                pickle.dump(reduced_features[phase], f)
    return reduced_features
