
import pickle
from urllib.parse import urlparse

import mlflow
import numpy as np
import pandas as pd

import src
from src.utils.ml import KmeansCluster


def get_clusterer(features, kmeans_args):
    clusterer = KmeansCluster.try_load()
    if clusterer is None:
        clusterer = KmeansCluster(kmeans_args)
        clusterer.train(features["train"])
    return clusterer


def cluster_all(clusterer, features):
    clustered = {}
    for phase in src.PHASES:
        clustered[phase] = clusterer.transform(features[phase])
    return clustered


def sort_cluster(clustered, labels, n_clusters):
    ca_counts = {phase: [] for phase in src.PHASES}
    ng_counts = {phase: [] for phase in src.PHASES}
    for phase in src.PHASES:
        df = pd.DataFrame(data={"label": labels[phase], "cluster": clustered[phase]})
        for c in range(n_clusters):
            _label = df["label"][df["cluster"] == c]
            n_ng = (_label == 0).sum()
            n_ca = (_label == 1).sum()
            #n_ca = (_label == 1).sum()
            ca_counts[phase].append(n_ca)
            ng_counts[phase].append(n_ng)

    n_ca = np.array(ca_counts["train"])
    n_ng = np.array(ng_counts["train"])
    
    ratio = n_ca/(n_ca+n_ng)
    new_map = np.argsort(ratio)

    sorted_clustered = {phase: [] for phase in src.PHASES}
    for phase in src.PHASES:
        for x in clustered[phase]:
            sorted_clustered[phase].append(np.argmax(new_map == x))
    return sorted_clustered


def make_hist(clustered, cumulative_sums, n_clusters):
    hists = {}
    for phase in src.PHASES:
        current = []
        cumulative_sum = [0]+cumulative_sums[phase]
        for i in range(len(cumulative_sum)-1):
            current.append(clustered[phase][cumulative_sum[i]:cumulative_sum[i+1]])

        def to_hist(line):
            hist = [0]*n_clusters
            for x in line:
                hist[x] += 1
            return hist
        hist = np.array(list(map(to_hist, current)))
        hists[phase] = hist
    return hists


def clusterize_features(features, labels, kmeans_args):
    n_clusters = kmeans_args.n_clusters
    clusterer = get_clusterer(features, kmeans_args)

    save_path = urlparse(mlflow.get_artifact_uri(f"clustered.pickle")).path
    try:
        print(f"trying load {save_path}...", end="")
        with open(save_path, "rb") as f:
            clustered = pickle.load(f)
        print("done")
    except:
        print("failed. Start transforming.")
        clustered = cluster_all(clusterer, features)
        clustered = sort_cluster(clustered, labels, n_clusters)
        with open(save_path, "wb") as f:
            pickle.dump(clustered, f)

    return clustered
