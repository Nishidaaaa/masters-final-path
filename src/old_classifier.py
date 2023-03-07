
from src.utils.ml import KmeansCluster, UmapReducer
import src
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import LogNorm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import umap
import time
import itertools
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from urllib.parse import urlparse
import json


def get_classifier(mode, config):
    if mode == "kmeans_gaussian":
        return KmeansGaussian(config)
    elif mode == "kmeans_svm":
        return KmeansSVM(config)
    elif mode == "kmeans_knn":
        return KmeansKNN(config)
    elif mode == "kmeans_rf":
        return KmeansRF(config)


def train_func(x):
    classifier, param, normals, labels = x
    classifier = classifier(**param)
    classifier.fit(normals["train"], labels["train"])
    preds = {phase: classifier.predict(normals[phase]) for phase in src.PHASES}
    accs = {phase: accuracy_score(labels[phase], preds[phase]) for phase in src.PHASES}
    return [accs, param]


class HistgramKmeans:
    def __init__(self, kmeans_args) -> None:
        self.kmeans_args = kmeans_args

    def _clustering(self, preprocessed):
        self.cluster = KmeansCluster.try_load()
        if self.cluster is None:
            self.cluster = KmeansCluster(self.kmeans_args)
            self.cluster.train(preprocessed["train"])
        self.clustered = {}
        for phase in src.PHASES:
            self.clustered[phase] = self.cluster.transform(preprocessed[phase])

    def _sort_cluster(self, datasets):
        labels = {phase: datasets[phase].labels for phase in src.PHASES}
        ca_counts = {phase: [] for phase in src.PHASES}
        ng_counts = {phase: [] for phase in src.PHASES}
        for phase in src.PHASES:
            df = pd.DataFrame(data={"label": labels[phase], "cluster": self.clustered[phase]})
            for c in range(self.kmeans_args.n_clusters):
                _label = df["label"][df["cluster"] == c]
                n_ng = (_label == 0).sum()
                n_ca = (_label == 1).sum()
                ca_counts[phase].append(n_ca)
                ng_counts[phase].append(n_ng)
            # fig, ax = plt.subplots(figsize=(10, 8))
            # counts = pd.DataFrame([ca_counts[phase], ng_counts[phase]], index=["ca", "ng"]).T
            # counts.plot(kind='bar', stacked=True, ax=ax, color=["red", "blue"])
            # mlflow.log_figure(fig, f"counts_{phase}.png")
            # plt.close('all')

        n_ca = np.array(ca_counts["train"])
        n_ng = np.array(ng_counts["train"])

        ratio = n_ca/(n_ca+n_ng)
        new_map = np.argsort(ratio)
        new_cluster = {phase: [] for phase in src.PHASES}
        for phase in src.PHASES:
            for x in self.clustered[phase]:
                new_cluster[phase].append(np.argmax(new_map == x))
        self.clustered = new_cluster

    def _make_hist(self, datasets):
        self.hists = {}
        for phase in src.PHASES:
            current = []
            dataset = datasets[phase]
            cumulative_sum = [0]+dataset.cumulative_sum
            for i in range(len(cumulative_sum)-1):
                current.append(self.clustered[phase][cumulative_sum[i]:cumulative_sum[i+1]])

            def to_hist(line):
                hist = [0]*self.kmeans_args.n_clusters
                for x in line:
                    hist[x] += 1
                return hist
            hist = np.array(list(map(to_hist, current)))
            self.hists[phase] = hist
        return self.hists

    def fit(self, datasets, preprocessed):
        self._clustering(preprocessed)
        self._sort_cluster(datasets)
        self._make_hist(datasets)
        return self.clustered, self.hists


class Classifier:
    def __init__(self) -> None:
        pass


class KmeansGaussian(Classifier):
    def __init__(self, config):
        self.config = config
        self.hist_maker = HistgramKmeans(config.kmeans_args)
        self.n_clusters = config.kmeans_args.n_clusters

    def fit(self, datasets, preprocessed):
        self.clustered, self.hists = self.hist_maker.fit(datasets, preprocessed)
        self.analyze(datasets, preprocessed)
        self.predict(datasets)

    def analyze(self, datasets, preprocessed):
        def draw_scatter(x, y, c, title, s=2):
            fig = plt.figure(figsize=(12, 9))
            plt.scatter(x, y, c=c, cmap='Spectral', s=s)
            plt.colorbar(boundaries=np.arange(np.max(c)+2)-0.5).set_ticks(np.arange(np.max(c)+1))
            plt.title(title)
            mlflow.log_figure(fig, title+".png")
            plt.close('all')

        def draw_bar(n_clusters, labels, cluster, phase):
            ca_counts = {phase: [] for phase in src.PHASES}
            ng_counts = {phase: [] for phase in src.PHASES}
            df = pd.DataFrame(data={"label": labels[phase], "cluster": self.clustered[phase]})
            for c in range(n_clusters):
                _label = df["label"][df["cluster"] == c]
                n_ng = (_label == 0).sum()
                n_ca = (_label == 1).sum()
                ca_counts[phase].append(n_ca)
                ng_counts[phase].append(n_ng)
            fig, ax = plt.subplots(figsize=(10, 8))
            counts = pd.DataFrame([ca_counts[phase], ng_counts[phase]], index=["ca", "ng"]).T
            counts.plot(kind='bar', stacked=True, ax=ax, color=["red", "blue"])
            mlflow.log_figure(fig, f"counts2_{phase}.png")
            plt.close('all')
        labels = {phase: datasets[phase].labels for phase in src.PHASES}

        for phase in src.PHASES:
            draw_scatter(preprocessed[phase][:, 0], preprocessed[phase][:, 1], labels[phase], f"label_{phase}")
            draw_scatter(preprocessed[phase][:, 0], preprocessed[phase][:, 1], self.clustered[phase], f"cluster_{phase}")
            draw_bar(self.n_clusters, labels, self.clustered, phase)

    def predict(self, datasets):
        labels = {phase: datasets[phase].original_dataset.labels for phase in src.PHASES}
        normals = {phase: preprocessing.normalize(self.hists[phase]) for phase in src.PHASES}
        from scipy.spatial.distance import jensenshannon
        reducer = umap.UMAP(n_components=2, random_state=0, metric=jensenshannon)
        # reducer = PCA(n_components=2, random_state=0)
        reducer.fit(normals["train"])
        coods = {phase: reducer.transform(normals[phase]) for phase in src.PHASES}
        # coods = normals

        def draw_scatter(x, y, c, title, s=24):
            fig = plt.figure(figsize=(12, 9))
            plt.scatter(x, y, c=c, cmap='Spectral', s=s)
            plt.colorbar(boundaries=np.arange(np.max(c)+2)-0.5).set_ticks(np.arange(np.max(c)+1))
            plt.title(title)
            mlflow.log_figure(fig, title+".png")
            plt.close('all')

        for phase in ["train", "test"]:
            draw_scatter(coods[phase][:, 0], coods[phase][:, 1], labels[phase], f"{phase}_hists")

        best_ct = ""
        best_acc = -1
        for ct in ["full", "tied", "diag", "spherical"]:
            mixer = GaussianMixture(n_components=2, random_state=0, covariance_type=ct)
            mixer.fit(coods["train"], labels["train"])
            val_pred = mixer.predict(coods["val"])
            acc = accuracy_score(labels["val"], val_pred)
            if best_acc < acc:
                best_acc = acc
                best_ct = ct
        mixer = GaussianMixture(n_components=2, random_state=0, covariance_type=best_ct)
        mixer.fit(coods["train"])

        self.preds = {phase: mixer.predict(coods[phase]) for phase in src.PHASES}
        self.probas = {phase: mixer.predict_proba(coods[phase])[:, 1] for phase in src.PHASES}
        if accuracy_score(labels["train"], self.preds["train"]) < 0.5:
            for phase in src.PHASES:
                pred = np.array(self.preds[phase])
                self.preds[phase] = (1-pred).tolist()
                proba = np.array(self.probas[phase])
                self.probas[phase] = (1-proba).tolist()

        def draw_map(mixer, phase):
            x = np.linspace(-5, 15)
            y = np.linspace(-5, 15)
            X, Y = np.meshgrid(x, y)
            XX = np.array([X.ravel(), Y.ravel()]).T
            Z = -mixer.score_samples(XX)
            Z = Z.reshape(X.shape)
            fig, ax = plt.subplots(dpi=150, figsize=(5, 4))
            cs = ax.contourf(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                             levels=np.logspace(0, 3, 10), alpha=0.8)
            plt.colorbar(cs, shrink=0.95, label='Negative log-likelihood predicted')

            ax.scatter(coods[phase][labels[phase] == 0, 0], coods[phase][labels[phase] == 0, 1], marker='o', c='blue')
            ax.scatter(coods[phase][labels[phase] == 1, 0], coods[phase][labels[phase] == 1, 1], marker='x', c='red')

            mlflow.log_figure(fig, f"mixer/{phase}.png")

        for phase in src.PHASES:
            draw_map(mixer, phase)

        for phase in src.PHASES:
            acc = accuracy_score(labels[phase], self.preds[phase])
            print(phase, ":", acc)
            mlflow.log_metric(f"final_{phase}_acc", acc)


def rf_train_func(x):
    param, normals, labels = x
    classifier = RandomForestClassifier(**param)
    classifier.fit(normals["train"], labels["train"])
    preds = {phase: classifier.predict(normals[phase]) for phase in src.PHASES}
    accs = {phase: accuracy_score(labels[phase], preds[phase]) for phase in src.PHASES}
    return [accs, param]


class KmeansRF(KmeansGaussian):
    def predict(self, datasets):
        labels = {phase: datasets[phase].original_dataset.labels for phase in src.PHASES}
        normals = {phase: preprocessing.normalize(self.hists[phase]) for phase in src.PHASES}
        # normals = self.hists

        try:
            uri = mlflow.get_artifact_uri("classifier/bestparam.json")
            path = urlparse(uri).path
            with open(path, mode="rt") as f:
                best_param = json.load(f)
        except:
            grid = {
                "n_estimators": [10*i for i in range(1, 25, 2)],
                "criterion": ["gini", "entropy"],
                "max_depth": [None]+[i*10 for i in range(1, 25, 2)],
                "min_samples_split": [i*2 for i in range(1, 10, 2)],
                "min_samples_leaf": [i*2 for i in range(1, 10, 2)],
                "random_state": [0]
            }
            product = [x for x in itertools.product(*grid.values())]
            all_params = [dict(zip(grid.keys(), p)) for p in product]

            all_args = [(RandomForestClassifier, dict(zip(grid.keys(), p)), normals, labels) for p in product]

            with Pool(10) as pool:
                results = list(tqdm(pool.imap(train_func, all_args), total=len(all_args)))

            table = []
            for accs, param in results:
                table.append({**accs, **param})
            table = pd.DataFrame(table)
            table.to_csv("tmp/rfc.csv")
            mlflow.log_artifact("tmp/rfc.csv")

            best_val = -1
            best_param = {}

            for accs, param in results:
                if accs["val"] > best_val:
                    best_val = accs["val"]
                    best_param = param

        classifier = RandomForestClassifier(**best_param)
        mlflow.log_dict(best_param, "classifier/bestparam.json")
        classifier.fit(normals["train"], labels["train"])

        self.preds = {phase: classifier.predict(normals[phase]) for phase in src.PHASES}
        self.probas = {phase: classifier.predict_proba(normals[phase])[:, 1] for phase in src.PHASES}
        if accuracy_score(labels["train"], self.preds["train"]) < 0.5:
            for phase in src.PHASES:
                pred = np.array(self.preds[phase])
                self.preds[phase] = (1-pred).tolist()
                proba = np.array(self.probas[phase])
                self.probas[phase] = (1-proba).tolist()

        for phase in src.PHASES:
            acc = accuracy_score(labels[phase], self.preds[phase])
            print(phase, ":", acc)
            mlflow.log_metric(f"final_{phase}_acc", acc)

        print("importance:", classifier.feature_importances_)
        mlflow.log_text(str(classifier.feature_importances_), "importance.txt")


def svm_train_func(x):
    param, normals, labels = x
    classifier = SVC(**param)
    classifier.fit(normals["train"], labels["train"])
    preds = {phase: classifier.predict(normals[phase]) for phase in src.PHASES}
    accs = {phase: accuracy_score(labels[phase], preds[phase]) for phase in src.PHASES}
    return [accs, param]


class KmeansSVM(KmeansGaussian):

    def predict(self, datasets):
        labels = {phase: datasets[phase].original_dataset.labels for phase in src.PHASES}
        normals = {phase: preprocessing.normalize(self.hists[phase]) for phase in src.PHASES}
        grid = {"C": [10 ** i for i in range(-3, 2)],
                "kernel": ["linear", "rbf", "sigmoid"],
                "decision_function_shape": ["ovo", "ovr"],
                "random_state": [0]
                }
        product = [x for x in itertools.product(*grid.values())]
        all_args = [(SVC, dict(zip(grid.keys(), p)), normals, labels) for p in product]

        with Pool(10) as pool:
            results = list(tqdm(pool.imap(train_func, all_args), total=len(all_args)))
        table = []
        for accs, param in results:
            table.append({**accs, **param})
        table = pd.DataFrame(table)
        table.to_csv("tmp/svc.csv")
        mlflow.log_artifact("tmp/svc.csv")

        best_val = -1
        best_param = {}

        for accs, param in results:
            if accs["val"] > best_val:
                best_val = accs["val"]
                best_param = param

        classifier = SVC(**best_param, probability=True)
        classifier.fit(normals["train"], labels["train"])

        self.preds = {phase: classifier.predict(normals[phase]) for phase in src.PHASES}
        self.probas = {phase: classifier.predict_proba(normals[phase])[:, 1] for phase in src.PHASES}

        for phase in src.PHASES:
            acc = accuracy_score(labels[phase], self.preds[phase])
            print(phase, ":", acc)
            mlflow.log_metric(f"final_{phase}_acc", acc)


class KmeansKNN(KmeansGaussian):
    def predict(self, datasets):
        labels = {phase: datasets[phase].original_dataset.labels for phase in src.PHASES}
        normals = {phase: self.hists[phase] for phase in src.PHASES}

        # reducer = PCA(n_components=4, random_state=0)
        # reducer.fit(normals["train"])
        # coods = {phase: reducer.transform(normals[phase]) for phase in src.PHASES}
        coods = normals

        best_nn = -1
        best_acc = -1
        for nn in range(1, 20):
            mixer = KNeighborsClassifier(n_neighbors=nn)
            mixer.fit(coods["train"], labels["train"])
            val_pred = mixer.predict(coods["val"])
            acc = accuracy_score(labels["val"], val_pred)
            if best_acc < acc:
                best_acc = acc
                best_nn = nn

        mixer = KNeighborsClassifier(n_neighbors=best_nn)
        mixer.fit(coods["train"], labels["train"])

        self.preds = {phase: mixer.predict(coods[phase]) for phase in src.PHASES}
        for phase in src.PHASES:
            acc = accuracy_score(labels[phase], self.preds[phase])
            print(phase, ":", acc)
            mlflow.log_metric(f"final_{phase}_acc", acc)
