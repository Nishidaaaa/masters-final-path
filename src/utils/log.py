import pickle
import mlflow
from omegaconf import OmegaConf


def recover_config(original):
    try:
        uri = mlflow.get_artifact_uri("config/config.yaml")
        config = mlflow.artifacts.load_text(uri)
        return OmegaConf.create(config)
    except:
        return original


def log_config(config):
    config = OmegaConf.to_container(config, resolve=True)
    mlflow.log_dict(config, "config/config.yaml")
    log_params_recursive(config, "")


def log_params_recursive(dict_config, head):
    for k, v in dict_config.items():
        if isinstance(v, dict):
            log_params_recursive(v, head+k+"/")
        else:
            mlflow.log_param(head+k, v)


class IDLogger:
    def __init__(self) -> None:
        self.run_ids = {}

    def log(self, name):
        run_id = mlflow.active_run().info.run_id
        self.run_ids[name] = run_id
        mlflow.log_params(self.run_ids)
        mlflow.set_tag("run_id", run_id)
