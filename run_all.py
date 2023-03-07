import hydra
import mlflow
import omegaconf

from src.classify import train_rf_with_grid_search
from src.cluster import clusterize_features, make_hist
from src.extract_features import extract_features
from src.reduce_dimension import reduce_dimension
from src.utils.analyze import log_graphs
from src.utils.log import IDLogger, recover_and_log_config

experiment_name = "test"
# experiment_name = "run_all"


@hydra.main(config_path="./config", config_name="run_all", version_base="1.2")
def run_all(config: omegaconf.DictConfig) -> None:
    experiment = mlflow.set_experiment(experiment_name)
    id_logger = IDLogger()
    with mlflow.start_run(run_id=config.extract_features_id, experiment_id=experiment.experiment_id):
        id_logger.log("extract_features_id")
        extract_features_config = recover_and_log_config(config.extract_features)

        data_module, model, features = extract_features(experiment_name=experiment_name, **extract_features_config)

        patch_labels = data_module.patch_labels
        patch_classes = data_module.patch_classes
        cumulative_sums = data_module.cumulative_sums
        original_labels = data_module.original_labels
        original_classes = data_module.original_classes
        original_fullpaths = data_module.original_fullpaths

        with mlflow.start_run(run_id=config.reduce_dimension_id, nested=True):
            id_logger.log("reduce_dimension_id")
            reduce_dimension_config = recover_and_log_config(config.reduce_dimension)

            reduced_features = reduce_dimension(features=features, **reduce_dimension_config)

            with mlflow.start_run(run_id=config.clusterize_and_classify_id, nested=True):
                id_logger.log("clusterize_and_classify_id")
                clusterize_and_classify_config = recover_and_log_config(config.clusterize_and_classify)
                n_clusters = clusterize_and_classify_config.kmeans_args.n_clusters

                clustered_features = clusterize_features(reduced_features, patch_labels, clusterize_and_classify_config.kmeans_args)
                histgrams = make_hist(clustered_features, cumulative_sums, n_clusters)
                predictions = train_rf_with_grid_search(original_labels, histgrams)

                log_graphs(histgrams=histgrams, original_labels=original_labels, patch_classes=patch_classes,
                           patch_labels=patch_labels, clustered_features=clustered_features, reduced_features=reduced_features,
                           original_classes=original_classes, n_clusters=n_clusters, original_fullpaths=original_fullpaths,
                           predictions=predictions, cumulative_sums=cumulative_sums, image_size=extract_features_config.model.image_size)

                # try:
                #     save_attentions(model, current_config.classifier.kmeans_args.n_clusters,
                #                     np.array(classifier.clustered["train"]), datasets["train"])
                # except:
                #     print("no attentions provided.")


if __name__ == "__main__":
    run_all()
