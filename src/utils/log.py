import pickle
import mlflow
from omegaconf import OmegaConf, DictConfig


def recover_and_log_config(config: DictConfig) -> DictConfig:
    """ 実行中のrunにコンフィグファイルが保存されており, 復元可能な場合は復元したコンフィグを返します．復元できなかった場合は，引数で渡されたコンフィグを保存し，そのまま返します．

    Args:
        config (DictConfig): コンフィグが復元できなかった場合に，使用・保存するコンフィグ

    Returns:
        DictConfig: 復元できた場合は，復元したコンフィグ．できなかった場合は，引数で指定されたコンフィグ．
    """
    current_config = recover_config(config)
    log_config(current_config)
    return current_config


def recover_config(original: DictConfig) -> DictConfig:
    try:
        uri = mlflow.get_artifact_uri("config/config.yaml")
        config = mlflow.artifacts.load_text(uri)
        return OmegaConf.create(config)
    except:
        return original


def log_config(config: DictConfig) -> None:
    config = OmegaConf.to_container(config, resolve=True)
    mlflow.log_dict(config, "config/config.yaml")
    log_params_recursive(config, "")


def log_params_recursive(dict_config: DictConfig, head: str):
    for k, v in dict_config.items():
        if isinstance(v, dict):
            log_params_recursive(v, head+k+"/")
        else:
            mlflow.log_param(head+k, v)


class IDLogger:
    """runを入れ子にしたときに，親のrun idを保存するためのクラス
    """

    def __init__(self) -> None:
        self.run_ids = {}

    def log(self, name: str):
        """現在のrun id を指定された名前で現在のrunにlogします．また，過去に同インスタンスでlogされたrun idと名前も現在のrunにlogします．

        Args:
            name (str): run idの保存名
        """
        run_id = mlflow.active_run().info.run_id
        self.run_ids[name] = run_id
        mlflow.log_params(self.run_ids)
        mlflow.set_tag("run_id", run_id)
