from urllib.parse import urlparse
import mlflow
from sklearn.decomposition import PCA
import sklearn.preprocessing as preprocessing
import pickle
from sklearn.cluster import KMeans

import umap
import time


class MLWithBackup:
    def __init__(self) -> None:
        self.save_name = type(self).__name__
        pass

    def _train(self, train_codes):
        raise NotImplementedError()

    def _normalize(self, codes, should_normalize):
        if should_normalize:
            return preprocessing.normalize(codes)
        else:
            return codes

    def train(self, train_codes):
        print(f"start train {self.save_name}")
        start = time.time()
        self._train(train_codes)
        end = time.time()
        print(f"finish training [{end-start}]")
        save_path = f"tmp/{self.save_name}.pickle"
        with open(save_path, "wb") as f:
            pickle.dump(self, f, protocol=4)
        mlflow.log_artifact(save_path, "models")

    @classmethod
    def try_load(cls):
        try:
            # path = f"models/{cls.__name__}.pickle"
            uri = mlflow.get_artifact_uri(f"models/{cls.__name__}.pickle")
            path = urlparse(uri).path
            print(f"trying load {path}...", end="")
            with open(path, "rb") as f:
                loaded = pickle.load(f)
                print("done")
                return loaded
        except:
            print("failed")
            return None

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError()

    def transform(self, codes):
        raise NotImplementedError()


class PcaUmapReducer(MLWithBackup):
    def __init__(self, pca_args, umap_args, should_normalize=True) -> None:
        super().__init__()
        self.pca = PCA(**pca_args)
        self.umap = umap.UMAP(**umap_args)
        self.should_normalize = should_normalize

    def _train(self, train_codes):
        train_codes = self._normalize(train_codes, self.should_normalize)
        self.pca.fit(train_codes)
        after_pca = self.pca.transform(train_codes)
        self.umap.fit(after_pca)
        after_umap = self.umap.transform(after_pca)

    def transform(self, codes):
        codes = self._normalize(codes, self.should_normalize)
        after_pca = self.pca.transform(codes)
        after_umap = self.umap.transform(after_pca)
        return after_umap

    @classmethod
    def from_config(cls, config):
        return cls(config.pca_args, config.umap_args, config.should_normalize)


class UmapReducer(MLWithBackup):
    def __init__(self, umap_args, should_normalize) -> None:
        super().__init__()
        self.umap = umap.UMAP(**umap_args)
        self.should_normalize = should_normalize

    def _train(self, train_codes):
        train_codes = self._normalize(train_codes, self.should_normalize)##このへんかな
        print('train_codesのサイズは、' + str(len(train_codes)))
        self.umap.fit(train_codes)

    def transform(self, codes):
        codes = self._normalize(codes, self.should_normalize)
        return self.umap.transform(codes)

    @classmethod
    def from_config(cls, config):
        return cls(config.umap_args, config.should_normalize)


class PcaReducer(MLWithBackup):
    def __init__(self, pca_args, should_normalize) -> None:
        super().__init__()
        self.pca = PCA(**pca_args)
        self.should_normalize = should_normalize

    def _train(self, train_codes):
        train_codes = self._normalize(train_codes, self.should_normalize)
        self.pca.fit(train_codes)

    def transform(self, codes):
        codes = self._normalize(codes, self.should_normalize)
        return self.pca.transform(codes)

    @classmethod
    def from_config(cls, config):
        return cls(config.pca_args, config.should_normalize)


class KmeansCluster(MLWithBackup):
    def __init__(self, kmeans_args) -> None:
        super().__init__()
        self.kmeans = KMeans(**kmeans_args)
        self.should_normalize = False

    def _train(self, train_codes):
        train_codes = self._normalize(train_codes, self.should_normalize)
        self.kmeans.fit(train_codes)

    def transform(self, codes):
        codes = self._normalize(codes, self.should_normalize)
        return self.kmeans.predict(codes)

    @classmethod
    def from_config(cls, config):
        return cls(config.kmeans_args)
