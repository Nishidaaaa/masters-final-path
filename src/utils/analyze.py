import src
import pandas as pd
import seaborn as sns
import imagesize
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import mlflow
import sklearn.preprocessing as preprocessing
from umap import UMAP
from src.utils.data import ThumbnailDataset
import json
import japanize_matplotlib


def check_none(elements):
    for element in elements:
        if element is None:
            return False
    return True


def get_thumbnail(file, index):
    dataset = ThumbnailDataset(dataframe=file, p=10)
    img, y = dataset[index]
    return img


def log_heatmap(data_module, clustered, preds, n_clusters, image_size):
    for phase in src.PHASES:
        file = data_module.files[phase]
        dataset = data_module.patch_datasets[phase]
        pred = preds[phase]
        for index, row in file.iterrows():
            w, h = imagesize.get(row["fullpath"])
            w = w//image_size
            h = h//image_size
            right = dataset.cumulative_sum[index]
            left = ([0]+dataset.cumulative_sum)[index]
            clusters = np.array(clustered[phase][left:right])
            grid = np.reshape(clusters, [h, w])
            fig = plt.figure()

            ax = fig.add_subplot(2, 1, 1)
            img = get_thumbnail(file, index)
            ax.imshow(np.asarray(img))
            ax.axis("off")

            ax = fig.add_subplot(2, 1, 2)
            sns.heatmap(grid, vmax=n_clusters, vmin=0, square=True, ax=ax, cmap=sns.diverging_palette(
                center="dark", h_neg=240, h_pos=0, n=16, l=60, s=75))
            ax.axis("off")

            mlflow.log_figure(fig, f"heatmap/{phase}/{row['class']}_{row['label']}_{pred[index]}_{index}.png")
            plt.close()


def log_histgram(data_module, clustered, preds, n_clusters):
    for phase in src.PHASES:
        file = data_module.files[phase]
        dataset = data_module.patch_datasets[phase]
        pred = preds[phase]
        for index, row in file.iterrows():
            right = dataset.cumulative_sum[index]
            left = ([0]+dataset.cumulative_sum)[index]
            clusters = np.array(clustered[phase][left:right])
            histgram = [0]*n_clusters
            for c in clusters:
                histgram[c] += 1
            histgram = preprocessing.normalize([histgram])[0]
            fig = plt.figure()
            plt.bar(x=list(range(n_clusters)), height=histgram, width=1.0)
            plt.ylim((0.0, 1.0))
            plt.xticks(list(range(n_clusters)))

            plt.xlabel("クラスタ番号")
            plt.ylabel("頻度")
            mlflow.log_figure(fig, f"histgram/{phase}/{row['class']}_{row['label']}_{pred[index]}_{index}.pdf")
            plt.close()


def log_overview(data_module, preds, probas):
    for phase in src.PHASES:
        frame = data_module.files[phase]
        pred = preds[phase]
        proba = probas[phase]
        frame["pred"] = pred
        frame["correct"] = frame["pred"] == frame["label"]
        frame["proba"] = proba
        pd.options.display.float_format = "{:.3f}".format
        frame.to_csv(f"tmp/{phase}.csv")
        mlflow.log_artifact(f"tmp/{phase}.csv", "overview")
        correct = frame["correct"]
        print(f"{phase} accuracy :{sum(correct)}/{len(correct)}={sum(correct)/len(correct)}")
        class_group = frame.groupby(["class", "label"])
        print(class_group["correct"].agg(lambda x: f"{sum(x)}/{len(x)}={sum(x)/len(x)}"))


def log_cluster_scatter(data_module, preprocessed, clustered):
    for phase in src.PHASES:
        file = data_module.files[phase]

        x = np.array(preprocessed[phase][:, 0])
        y = np.array(preprocessed[phase][:, 1])
        labels = np.array(data_module.patch_datasets[phase].labels)
        _class = np.array(data_module.patch_datasets[phase].classes)
        claster = np.array(clustered[phase])
        idx = np.array(range(len(x)))
        np.random.shuffle(idx)
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        colors = np.array(["red" if label else "blue" for label in labels])
        plt.scatter(x[idx], y[idx], c=colors[idx], s=1)
        plt.xlabel("umap_1")
        plt.ylabel("umap_2")
        mlflow.log_figure(fig, f"scatter/label_{phase}.png")
        plt.close()

        # c = 0
        # for current_class in np.unique(_class):
        #     index = (_class == current_class)
        #     plt.scatter(x[index], y[index], c=[c]*len(x[index]), label=current_class, s=4)
        #     c += 1
        # plt.legend()
        # ax = fig.add_subplot(3, 1, 3)
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        plt.scatter(x[idx], y[idx], c=claster[idx], cmap='Spectral', s=2)
        plt.colorbar(boundaries=np.arange(np.max(claster)+2)-0.5).set_ticks(np.arange(np.max(claster)+1))
        plt.xlabel("umap_1")
        plt.ylabel("umap_2")
        mlflow.log_figure(fig, f"scatter/claster_{phase}.png")
        plt.close()


def log_final_scatter(data_module, hists):
    from scipy.spatial.distance import jensenshannon, euclidean
    # reducer = umap.UMAP(n_components=2, random_state=0, metric=jensenshannon)

    for metric in [jensenshannon, euclidean]:
        umap = UMAP(n_components=2, random_state=0, metric=metric)
        umap.fit(hists["train"])
        coods = {phase: umap.transform(hists[phase]) for phase in src.PHASES}
        for phase in src.PHASES:
            file = data_module.files[phase]
            x = np.array(coods[phase][:, 0])
            y = np.array(coods[phase][:, 1])
            labels = np.array(file["label"])
            _class = np.array(file["class"])

            fig = plt.figure(figsize=(16/2.54, 10/2.54))
            for label in [True, False]:
                index = (labels == label)
                plt.scatter(x[index], y[index], c="red" if label else "blue", cmap="Spectral", s=8)

            plt.xlabel("umap_1")
            plt.ylabel("umap_2")
            mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/label_{phase}.png")
            plt.close()

            fig = plt.figure(figsize=(16/2.54, 10/2.54))
            for current_class in np.unique(_class):
                for label in [True, False]:
                    index = (_class == current_class) & (labels == label)
                    plt.scatter(x[index], y[index], c="red" if label else "blue", cmap="Spectral",
                                marker=f"${current_class}$", s=64)
            plt.xlabel("umap_1")
            plt.ylabel("umap_2")
            mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/class_{phase}.pdf")
            plt.close()

        fig = plt.figure(figsize=(16/2.54, 10/2.54))
        for phase in src.PHASES:
            file = data_module.files[phase]
            x = np.array(coods[phase][:, 0])
            y = np.array(coods[phase][:, 1])
            labels = np.array(file["label"])
            _class = np.array(file["class"])

            for label in [True, False]:
                index = (labels == label)
                plt.scatter(x[index], y[index], c="red" if label else "blue", cmap="Spectral", s=8)
        plt.xlabel("umap_1")
        plt.ylabel("umap_2")
        mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/label_all.pdf")
        plt.close()

        fig = plt.figure(figsize=(16/2.54, 10/2.54))
        for phase in src.PHASES:
            file = data_module.files[phase]
            x = np.array(coods[phase][:, 0])
            y = np.array(coods[phase][:, 1])
            labels = np.array(file["label"])
            _class = np.array(file["class"])
            for current_class in np.unique(_class):
                for label in [True, False]:
                    index = (_class == current_class) & (labels == label)
                    plt.scatter(x[index], y[index], c="red" if label else "blue", cmap="Spectral", marker=f"${current_class}$", s=64)
        mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/class_all.pdf")
        plt.close()


def log_cluster_bar(data_module, clustered, n_clusters):
    for phase in src.PHASES:
        cluster = np.array(clustered[phase])
        labels = np.array(data_module.patch_datasets[phase].labels)

        counts = {}
        bar_labels = []
        for c in range(n_clusters):
            current = cluster == c
            count = []
            for label in [0, 1]:
                count.append((labels[current] == label).sum())
            counts[c] = count
            bar_labels.append(f"{count[1]/(sum(count))*100:.0f}%")

        frame = pd.DataFrame.from_dict(counts, columns=["正常", "がん"], orient="index")
        fig, ax = plt.subplots(figsize=(16/2.54, 10/2.54))
        b1 = ax.bar(list(frame.index), frame["正常"], color="blue", label="正常")
        b2 = ax.bar(list(frame.index), frame["がん"], color="red", label="がん", bottom=frame["正常"])
        plt.xlabel("クラスタ番号")
        plt.ylabel("パッチ画像枚数")
        plt.legend()
        plt.xticks(list(range(n_clusters)))
        ax.bar_label(b2, labels=bar_labels, label_type="edge")
        mlflow.log_figure(fig, f"cluster_bar_label_proto/{phase}.pdf")

    #     fig, ax = plt.subplots(figsize=(16/2.54, 10/2.54))
    #     frame.plot(kind='bar', stacked=True, ax=ax, color=["blue", "red"])
    #     plt.xlabel("クラスタ番号")
    #     plt.ylabel("パッチ画像枚数")

    #     xxx = 0
    #     for c in ax.containers:
    #         try:
    #             ax.bar_label(c, labels=bar_labels[xxx], label_type='center')
    #             xxx += 1
    #         except:
    #             pass
    #     # ax.bar_label(ax.containers[0], labels=bar_labels, label_type='center')
    #     mlflow.log_figure(fig, f"cluster_bar_label/{phase}.pdf")
    #     plt.close('all')
    # for phase in src.PHASES:
    #     cluster = np.array(clustered[phase])
    #     labels = np.array(data_module.patch_datasets[phase].labels)
    #     _class = np.array(data_module.patch_datasets[phase].classes)
    #     counts = {}
    #     for c in range(n_clusters):
    #         current = cluster == c
    #         count = []
    #         for __class in np.unique(_class):
    #             count.append((_class[current] == __class).sum())
    #         counts[c] = count
    #     frame = pd.DataFrame.from_dict(counts, columns=np.unique(_class), orient="index")
    #     fig, ax = plt.subplots(figsize=(16/2.54, 10/2.54))
    #     frame.plot(kind='bar', stacked=True, ax=ax)
    #     plt.xlabel("クラスタ番号")
    #     plt.ylabel("パッチ画像枚数")
    #     mlflow.log_figure(fig, f"cluster_bar_class/{phase}.pdf")
        plt.close('all')


def log_globalheat(data_module, hists):
    for phase in src.PHASES:
        hist = hists[phase]
        hist = np.array(preprocessing.normalize(hist))
        file = data_module.files[phase]
        label = np.array(file["label"])

        fig, axes = plt.subplots(2, 1, figsize=(16/2.54, 10/2.54))
        plt.subplots_adjust(hspace=0.6)
        ax = axes[0]
        heatmap = ax.pcolor(hist[label == True], vmin=0, vmax=1.0, cmap="Reds")
        ax.set_title("がん")
        ax.set_xticks(np.arange(hist.shape[1]) + 0.5, np.arange(hist.shape[1]))
        fig.colorbar(heatmap, ax=ax)
        ax.set_xlabel("クラスタ番号")
        ax.set_ylabel("画像番号")

        ax = axes[1]
        heatmap = ax.pcolor(hist[label == False], vmin=0, vmax=1.0, cmap="Reds")
        ax.set_title("正常")
        ax.set_xticks(np.arange(hist.shape[1]) + 0.5, np.arange(hist.shape[1]))
        fig.colorbar(heatmap, ax=ax)
        ax.set_xlabel("クラスタ番号")
        ax.set_ylabel("画像番号")

        mlflow.log_figure(fig, f"global_heat/{phase}.pdf")
        plt.close('all')

    



def analyze(data_module=None, preprocessed=None,  preds=None, hists=None, probas=None, clustered=None, n_clusters=None, image_size=None):
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelpad"] = 2
    log_globalheat(data_module, hists)
    log_cluster_scatter(data_module, preprocessed, clustered)
    log_final_scatter(data_module, hists)
    log_cluster_bar(data_module, clustered, n_clusters)

    log_heatmap(data_module, clustered, preds, n_clusters, image_size)
    log_histgram(data_module, clustered, preds, n_clusters)
    log_overview(data_module, preds, probas)
