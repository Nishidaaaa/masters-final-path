import os
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
from statistics import stdev, variance, median


def check_none(elements):
    for element in elements:
        if element is None:
            return False
    return True


def get_thumbnail(fullpaths, index):
    dataset = ThumbnailDataset(fullpaths=fullpaths, p=10)
    img = dataset[index]
    return img

#ok
def log_heatmap(clustered_features, predictions, n_clusters, image_size, original_fullpaths, cumulative_sums, original_classes, original_labels, pairs):
    for phase in src.PHASES:
        # for index, row in file.iterrows():
        cumulative_sum = cumulative_sums[phase]
        for index, (fullpath, _class, label, pred) in enumerate(zip(original_fullpaths[phase], original_classes[phase], original_labels[phase], predictions[phase])):
            w, h = imagesize.get(fullpath)
            w = w//image_size
            h = h//image_size
            right = cumulative_sum[index]
            left = ([0]+cumulative_sum)[index]
            clusters = np.array(clustered_features[phase][left:right])
            grid = np.reshape(clusters, [h, w])
            fig = plt.figure()

            ax = fig.add_subplot(2, 1, 1)
            img = get_thumbnail(original_fullpaths[phase], index)
            ax.imshow(np.asarray(img))
            ax.axis("off")

            ax = fig.add_subplot(2, 1, 2)
            sns.heatmap(grid, vmax=n_clusters, vmin=0, square=True, ax=ax, cmap=sns.diverging_palette(
                center="dark", h_neg=240, h_pos=0, n=16, l=60, s=75))
            ax.axis("off")
            name = os.path.splitext(os.path.basename(fullpath))[0]

            mlflow.log_figure(fig, f"heatmap/{phase}/{_class}_{label}_{pred}_{name}.png")
            plt.close()


def log_histgram(original_fullpaths, original_classes, original_labels, predictions, n_clusters, histgrams, pairs, confidences):
    for i in range(len(pairs)):
        new_name1 = 'correct_' + str(i) #collect_1
        new_name2 = 'woring_' + str(i)
        globals()[new_name1] = []
        globals()[new_name2] = []
    classes = []
    results = []
    confs =[]
    f = open('pred_history.txt', 'w')
    for phase in src.PHASES:
        #normalized_histgram = preprocessing.normalize(histgrams[phase])
        normalized_histgram = histgrams[phase]

        file_infos = zip(original_fullpaths[phase], normalized_histgram, predictions[phase], original_classes[phase], original_labels[phase], confidences[phase])
        for fullpath, histgram, pred, _class, label, conf in file_infos:
            fig = plt.figure()
            plt.bar(x=list(range(n_clusters)), height=histgram/sum(histgram), width=1.0)
            plt.ylim((0.0, 1.0))
            plt.xticks(list(range(n_clusters)))
            plt.xlabel("クラスタ番号")
            plt.ylabel("頻度")
            name = os.path.splitext(os.path.basename(fullpath))[0]
            mlflow.log_figure(fig, f"histgram/{phase}/{_class}_{label}_{pred}_{name}.pdf")
            plt.close()
            
            filename = os.path.basename(fullpath)
            f.write(filename + ' ---> ' + str(pred) + ' with conf of ' + str(round(conf, 4)) + '  ')
            f.write('o' if pred == label else 'x')
            f.write('\n')
            classes.append(label)
            results.append(True if pred == label else False)
            confs.append(round(conf, 4))
    f.close()        

    hist_correct = []
    hist_wrong = []
    std_correct = []
    std_wrong = []
    for i in range(len(pairs)):
        data_correct =[]
        data_wrong =[]
        for j in range(len(classes)):
            if classes[j] == i and results[j] == True:
                data_correct.append(confs[j])
            elif classes[j] == i and results[j] == False:
                data_wrong.append(confs[j])
        std_correct.append(0 if len(data_correct) <= 1 else stdev(data_correct))
        std_wrong.append(0 if len(data_wrong) <= 1 else stdev(data_wrong))
        hist_correct.append(np.mean(data_correct) if len(data_correct) != 0 else 0)
        hist_wrong.append(np.mean(data_wrong) if len(data_wrong) != 0 else 0)
            
    #print(hist_correct)
    #print(hist_wrong)
    
    fig = plt.figure()
    height = np.array(hist_correct)
    plt.bar(x=range(len(pairs)), height = height, yerr=std_correct, align="center", capsize=8, ecolor="red")
    plt.xlabel("ラベル")
    plt.ylabel("確信度")
    mlflow.log_figure(fig, f"confidence/correct.pdf")
    plt.close()

    fig = plt.figure()
    height = np.array(hist_wrong)
    plt.bar(x=range(len(pairs)), height = height, yerr=std_wrong, align="center",  capsize=8, ecolor="red")
    plt.xlabel("ラベル")
    plt.ylabel("確信度")
    mlflow.log_figure(fig, f"confidence/wrong.pdf")
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


def log_cluster_scatter(patch_labels, patch_classes, reduced_features, clusterd_features, pairs):
    for phase in src.PHASES:

        x = np.array(reduced_features[phase][:, 0])
        y = np.array(reduced_features[phase][:, 1])
        labels = np.array(patch_labels[phase])
        _class = np.array(patch_classes[phase])
        cluster = np.array(clusterd_features[phase])
        '''
        idx = np.array(range(len(x)))
        np.random.shuffle(idx)
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        colors = np.array(["red" if label else "blue" for label in labels])
        plt.scatter(x[idx], y[idx], c=colors[idx], s=1)
        plt.xlabel("umap_1")
        plt.ylabel("umap_2")
        mlflow.log_figure(fig, f"scatter/label_{phase}.png")
        plt.close()
        '''
        idx = np.array(range(len(x)))
        np.random.shuffle(idx)
        fig = plt.figure(figsize=(10, 5), tight_layout=True)
        #colors = np.array(["red" if label else "blue" for label in labels])
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = np.array([cycle[label] for label in labels])
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
        plt.scatter(x[idx], y[idx], c=cluster[idx], cmap='Spectral', s=2)
        plt.colorbar(boundaries=np.arange(np.max(cluster)+2)-0.5).set_ticks(np.arange(np.max(cluster)+1))
        plt.xlabel("umap_1")
        plt.ylabel("umap_2")
        mlflow.log_figure(fig, f"scatter/claster_{phase}.png")
        plt.close()


def log_final_scatter(original_labels, original_classes, histgrams, pairs):
    from scipy.spatial.distance import jensenshannon, euclidean
    # reducer = umap.UMAP(n_components=2, random_state=0, metric=jensenshannon)

    for metric in [jensenshannon, euclidean]:
        umap = UMAP(n_components=2, random_state=0, metric=metric)
        umap.fit(histgrams["train"])#ココ
        coods = {phase: umap.transform(histgrams[phase]) for phase in src.PHASES}
        for phase in src.PHASES:
            x = np.array(coods[phase][:, 0])
            y = np.array(coods[phase][:, 1])
            labels = np.array(original_labels[phase])
            _class = np.array(original_classes[phase])

            fig = plt.figure(figsize=(16/2.54, 10/2.54))
            #for label in [True, False]:
            for label in range(len(pairs)):
                index = (labels == label)
                cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                plt.scatter(x[index], y[index], c=cycle[label], s=8)

            plt.xlabel("umap_1")
            plt.ylabel("umap_2")
            mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/label_{phase}.png")
            plt.close()

            fig = plt.figure(figsize=(16/2.54, 10/2.54))
            for current_class in np.unique(_class):
                #for label in [True, False]:
                for label in range(len(pairs)):
                    index = (_class == current_class) & (labels == label)
                    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                    plt.scatter(x[index], y[index], c=cycle[label],
                                marker=f"${current_class}$", s=64)
            plt.xlabel("umap_1")
            plt.ylabel("umap_2")
            mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/class_{phase}.pdf")
            plt.close()

        fig = plt.figure(figsize=(16/2.54, 10/2.54))
        for phase in src.PHASES:
            x = np.array(coods[phase][:, 0])
            y = np.array(coods[phase][:, 1])
            labels = np.array(original_labels[phase])
            _class = np.array(original_classes[phase])

            for label in range(len(pairs)):
            #for label in [True, False]:
                index = (labels == label)
                cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
                plt.scatter(x[index], y[index], c=cycle[label], s=8)
        plt.xlabel("umap_1")
        plt.ylabel("umap_2")
        mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/label_all.pdf")
        plt.close()

        fig = plt.figure(figsize=(16/2.54, 10/2.54))
        for phase in src.PHASES:
            x = np.array(coods[phase][:, 0])
            y = np.array(coods[phase][:, 1])
            labels = np.array(original_labels[phase])
            _class = np.array(original_classes[phase])
            for current_class in np.unique(_class):
                #for label in [True, False]:
                for label in range(len(pairs)):
                    index = (_class == current_class) & (labels == label)
                    plt.scatter(x[index], y[index], c=cycle[label], marker=f"${current_class}$", s=64)
        mlflow.log_figure(fig, f"final_scatter/{metric.__name__}/class_all.pdf")
        plt.close()

#ok
def log_cluster_bar(patch_labels, clustered_features, n_clusters, pairs):
    for phase in src.PHASES:
        cluster = np.array(clustered_features[phase])
        labels = np.array(patch_labels[phase])

        counts = {}
        bar_labels = []
        
        for c in range(n_clusters):
            current = cluster == c
            count = []
            for i in range(len(pairs)):
            #for label in [0, 1]:
                count.append((labels[current] == i).sum())
            counts[c] = count
            bar_labels.append(f"{count[1]/(sum(count))*100:.0f}%")
        #for column in range(len(pairs)):
        frame = pd.DataFrame.from_dict(counts,orient = "index")
        fig, ax = plt.subplots(figsize=(24/2.54, 16/2.54))
        #cmap = plt.get_cmap("tab10")
        cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        labels = []
        #ax.bar(list(frame.index), frame[0], color=cycle[0], label=pairs[0]['name'])
        ax.bar(list(frame.index), frame[0], color=cycle[0])
        labels.append(pairs[0]['name'])
        bottoms = frame[0] 
        for i in range(len(pairs)-1):
            #ax.bar(list(frame.index), frame[i+1], color=cycle[i+1], label=pairs[i+1]['name'], bottom=bottoms)
            ax.bar(list(frame.index), frame[i+1], color=cycle[i+1], bottom=bottoms)
            bottoms += frame[i+1]
            labels.append(pairs[i+1]['name'])
            
        plt.xlabel("クラスタ番号")
        plt.ylabel("パッチ画像枚数")
        #plt.legend(labels, loc='lower left', bbox_to_anchor=(0, -0.4))s
        plt.legend()
        plt.xticks(list(range(n_clusters)))
        #ax.bar_label(b2, labels=bar_labels, label_type="edge")
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

def my_log_cluster_bar(patch_labels, clustered_features, n_clusters):
    for phase in src.PHASES:
        cluster = np.array(clustered_features[phase])
        labels = np.array(patch_labels[phase])

        counts = {}
        bar_labels = []
        for c in range(n_clusters):
            current = cluster == c
            count = []
            for label in [0, 3]:
            #for label in [0, 1]:
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

#ok
def log_global_heat(original_labels, histgrams, pairs):

    for phase in src.PHASES:
        hist = histgrams[phase]
        hist = np.array(preprocessing.normalize(hist))
        label = np.array(original_labels[phase])
        #print("pairs.size() : " + str(len(pairs)))
        #print(type(len(pairs)))
        fig, axes = plt.subplots(len(pairs), 1, figsize=(24/2.54, len(pairs)*6/2.54))
        plt.subplots_adjust(hspace=1.0)
        
        for i in range(len(pairs)):
            ax = axes[i]
            heatmap = ax.pcolor(hist[label == i], vmin=0, vmax=1.0, cmap="Reds")
            ax.set_title(pairs[i]["name"])
            ax.set_xticks(np.arange(hist.shape[1]) + 0.5, np.arange(hist.shape[1]))
            fig.colorbar(heatmap, ax=ax)
            ax.set_xlabel("クラスタ番号")
            ax.set_ylabel("画像番号")

        mlflow.log_figure(fig, f"global_heat/{phase}.pdf")
        plt.close('all')


# def analyze(data_module=None, preprocessed=None,  preds=None, hists=None, probas=None, clustered=None, n_clusters=None, image_size=None):
#     plt.rcParams["font.size"] = 10
#     plt.rcParams["axes.labelpad"] = 2
#     log_globalheat(data_module, hists)
#     log_cluster_scatter(data_module, preprocessed, clustered)
#     log_final_scatter(data_module, hists)
#     log_cluster_bar(data_module, clustered, n_clusters)

#     log_heatmap(data_module, clustered, preds, n_clusters, image_size)
#     log_histgram(data_module, clustered, preds, n_clusters)
#     log_overview(data_module, preds, probas)


def log_graphs(histgrams, original_labels, patch_labels, patch_classes, clustered_features, reduced_features, original_classes, n_clusters, predictions, original_fullpaths, cumulative_sums, image_size, pairs, confidences):
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelpad"] = 2
    log_global_heat(histgrams=histgrams, original_labels=original_labels, pairs=pairs)

    log_cluster_scatter(patch_labels=patch_labels, patch_classes=patch_classes,
                        clusterd_features=clustered_features, reduced_features=reduced_features, pairs=pairs)

    #log_final_scatter(original_labels=original_labels, original_classes=original_classes, histgrams=histgrams, pairs=pairs)

    log_cluster_bar(clustered_features=clustered_features, n_clusters=n_clusters, patch_labels=patch_labels, pairs=pairs)

    log_histgram(histgrams=histgrams, n_clusters=n_clusters, original_classes=original_classes,original_fullpaths=original_fullpaths, 
                 original_labels=original_labels, predictions=predictions, pairs=pairs, confidences = confidences)

    log_heatmap(clustered_features=clustered_features, predictions=predictions, n_clusters=n_clusters, image_size=image_size,original_fullpaths=original_fullpaths, 
                cumulative_sums=cumulative_sums, original_classes=original_classes, original_labels=original_labels, pairs=pairs)
    # log_overview
