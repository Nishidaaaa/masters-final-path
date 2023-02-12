from src.utils.ml import PcaUmapReducer, UmapReducer, PcaReducer
from urllib.parse import urlparse
import mlflow
import pickle
import src


def preprocess(config, codes):
    # mlflow.log_params(config)
    methods = {"pca": PcaReducer, "pca_umap": PcaUmapReducer, "umap": UmapReducer}
    method = methods[config.mode]
    reducer = method.try_load()
    if reducer is None:
        reducer = method.from_config(config.reducer)
        reducer.train(codes["train"])

    fcodes = {}
    for phase in src.PHASES:
        save_path = urlparse(mlflow.get_artifact_uri(f"{phase}.pickle")).path
        try:
            print(f"trying load {save_path}...", end="")
            with open(save_path, "rb") as f:
                fcodes[phase] = pickle.load(f)
            print("done")
        except:
            print("failed. Start transforming.")
            fcodes[phase] = reducer.transform(codes[phase])
            with open(save_path, "wb") as f:
                pickle.dump(fcodes[phase], f)

    return fcodes
