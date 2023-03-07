

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


def train_func(x):
    classifier, param, normals, labels = x
    classifier = classifier(**param)
    classifier.fit(normals["train"], labels["train"])
    preds = {phase: classifier.predict(normals[phase]) for phase in src.PHASES}
    accs = {phase: accuracy_score(labels[phase], preds[phase]) for phase in src.PHASES}
    return [accs, param]


def train_rf_with_grid_search(labels, histgrams,):
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

        all_args = [(RandomForestClassifier, dict(zip(grid.keys(), p)), histgrams, labels) for p in product]

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
    classifier.fit(histgrams["train"], labels["train"])

    preds = {phase: classifier.predict(histgrams[phase]) for phase in src.PHASES}
    probas = {phase: classifier.predict_proba(histgrams[phase])[:, 1] for phase in src.PHASES}
    if accuracy_score(labels["train"], preds["train"]) < 0.5:
        for phase in src.PHASES:
            pred = np.array(preds[phase])
            preds[phase] = (1-pred).tolist()
            proba = np.array(probas[phase])
            probas[phase] = (1-proba).tolist()

    for phase in src.PHASES:
        acc = accuracy_score(labels[phase], preds[phase])
        print(phase, ":", acc)
        mlflow.log_metric(f"final_{phase}_acc", acc)

    print("importance:", classifier.feature_importances_)
    mlflow.log_text(str(classifier.feature_importances_), "importance.txt")

    return preds
