import hydra
import mlflow
import omegaconf

from src.classify import train_rf_with_grid_search
from src.cluster import clusterize_features, make_hist
from src.extract_features import extract_features
from src.reduce_dimension import reduce_dimension
from src.utils.analyze import log_graphs
from src.utils.log import IDLogger, recover_and_log_config


# experiment_name = "run_all"


@hydra.main(config_path="./config", config_name="run_all", version_base="1.2")
def run_all(config: omegaconf.DictConfig) -> None:
    """提案手法をすべて実行します．
    各種解析結果やグラフも出力します．
    mlflowを3段の入れ子にして実験データを管理しています．
    これにより，訓練済みのモデルや出力結果を異なる条件で実行する際に再利用することができます．
    また，コンフィグはHydraによって管理しています．

    mlflowはexperiment(フォルダ)にrun(実験結果ファイル)を入れていくようなイメージでデータを管理します．
    各runはrun_idという固有のidを持ちます．
    実験データを再利用する際は，そのrun_idをコンフィグで指定してください．
    指定されなかった場合は，ランダムにidが割り振られて，新しく訓練を行います．
    *******3段構成ですが，再利用する際は必ず親のデータも再利用してください．******

    Args:
        config (omegaconf.DictConfig): 一連の実行に関するコンフィグ. デコレータによって自動で渡されます．
    """
    experiment_name = config.experiment_name
    experiment = mlflow.set_experiment(experiment_name)
    id_logger = IDLogger()

    with mlflow.start_run(run_id=config.extract_features_id, experiment_id=experiment.experiment_id):
        """mlflow1段目
        パッチ画像から自己教師で特徴を抽出するDINOを訓練します．
        モデルや抽出した特徴量等...は保存されます．
        もしすでに保存されていた場合は読み込みます．
        """
        id_logger.log("extract_features_id")
        extract_features_config = recover_and_log_config(config.extract_features)

        data_module, model, features = extract_features(experiment_name=experiment_name, **extract_features_config)

        # パッチレベルのデータ(サンプル数大)
        patch_labels = data_module.patch_labels
        patch_classes = data_module.patch_classes

        # 元画像レベルのデータ(サンプル数小)
        original_labels = data_module.original_labels
        original_classes = data_module.original_classes
        original_fullpaths = data_module.original_fullpaths
        # 各画像から何枚パッチを生成したかを累積和にしたもの
        # 例 画像3枚でそれぞれ30枚40枚30枚のパッチを切り出した場合
        #    cumulative_sums=[30, 70, 100]
        cumulative_sums = data_module.cumulative_sums

        with mlflow.start_run(run_id=config.reduce_dimension_id, nested=True):
            """mlflow2段目
            1段目で抽出した特徴量を次元圧縮します．
            次元圧縮に用いたモデルや次元圧縮後のデータは保存されます．
            もしすでに保存されていた場合は読み込みます．
            """
            #print("hello1")
            id_logger.log("reduce_dimension_id")
            #print("hello2")
            reduce_dimension_config = recover_and_log_config(config.reduce_dimension)
            #print("hello3")
            reduced_features = reduce_dimension(features=features, **reduce_dimension_config)
            #print("hello4")
            with mlflow.start_run(run_id=config.clusterize_and_classify_id, nested=True):
                """mlflow3段目
                2段目で次元圧縮したデータから元画像のラベルを推論するようなモデルを訓練します．
                kmeansでクラスタリング->ヒストグラムに変形->RandomForestで推論という流れです．
                モデルや中間データは保存されます．
                もしすでに保存されていた場合は読み込みます．
                また，結果から様々なグラフを生成します．
                グラフは実行ごとに再生成されます．
                """
                id_logger.log("clusterize_and_classify_id")
                clusterize_and_classify_config = recover_and_log_config(config.clusterize_and_classify)
                n_clusters = clusterize_and_classify_config.kmeans_args.n_clusters

                clustered_features = clusterize_features(reduced_features, patch_labels, clusterize_and_classify_config.kmeans_args)
                histgrams = make_hist(clustered_features, cumulative_sums, n_clusters)
                predictions, probabilities, confidences = train_rf_with_grid_search(original_labels, histgrams)
                #print(probabilities)
                #print(predictions)
                #print(confidences)
                log_graphs(histgrams=histgrams, original_labels=original_labels, patch_classes=patch_classes,
                           patch_labels=patch_labels, clustered_features=clustered_features, reduced_features=reduced_features,
                           original_classes=original_classes, n_clusters=n_clusters, original_fullpaths=original_fullpaths,
                           predictions=predictions, cumulative_sums=cumulative_sums, confidences = confidences, 
                           image_size=extract_features_config.model.image_size, pairs=extract_features_config.data_module.filemanager.pairs)

                # try:
                #     save_attentions(model, current_config.classifier.kmeans_args.n_clusters,
                #                     np.array(classifier.clustered["train"]), datasets["train"])
                # except:
                #     print("no attentions provided.")


if __name__ == "__main__":
    run_all()
