import pandas as pd
from PIL import Image
import os
from torch.utils.data import Dataset
import re
import glob
import numpy as np
import imagesize
import random
Image.MAX_IMAGE_PIXELS = 1000000000


class PatchDataset(Dataset):
    def __init__(self, dataframe, patch_size, transforms):
        self._n_patches = 0
        self._cache_index = -1
        self.cumulative_sum = []
        self.dataframe = dataframe
        self.patch_size = patch_size
        self.transforms = transforms
        self._labels = []
        self._classes = []
        self.original_dataset = OriginalImageDataset(dataframe, None)
        for path in dataframe["fullpath"]:
            width, height = imagesize.get(path)
            self._n_patches = self._n_patches+(width//patch_size)*(height//patch_size)
            self.cumulative_sum.append(self._n_patches)

    def _get_original_index(self, index):
        for i, csum in enumerate(self.cumulative_sum):
            if index < csum:
                return i

    def _get_patch_coods(self, index):
        original_image_index = self._get_original_index(index)
        patch_id = index-([0]+self.cumulative_sum)[original_image_index]
        width, height = imagesize.get(self.dataframe["fullpath"][original_image_index])
        v_patches = width//self.patch_size
        h_patches = height//self.patch_size

        x = patch_id % v_patches
        y = patch_id//v_patches
        return x, y

    def _get_original_image(self, original_image_index):
        if self._cache_index != original_image_index:
            self._cache_index = original_image_index
            self._cache_image, _ = self.original_dataset[original_image_index]

        return self._cache_image

    def _get_patch_image(self, original_image_index, x, y):
        original_image = self._get_original_image(original_image_index)
        X = x*self.patch_size
        Y = y*self.patch_size
        return original_image.crop((X, Y, X+self.patch_size, Y+self.patch_size))

    def get_cache_path(self, original_image_index, x, y):
        basename = self.dataframe["basename"][original_image_index]
        return f"./dataset/cache/wsidataset/{self.patch_size}/{basename}.{x:0>3}.{y:0>3}.png"

    def __getitem__(self, index):
        original_image_index = self._get_original_index(index)
        x, y = self._get_patch_coods(index)
        cache_path = self.get_cache_path(original_image_index, x, y)
        if os.path.exists(cache_path):
            patch_image = Image.open(cache_path)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            patch_image = self._get_patch_image(original_image_index, x, y)
            patch_image.save(cache_path)

        patch_image = self.transforms(patch_image)
        label = self.dataframe["label"][original_image_index]
        return patch_image, label

    def get_image_without_transform(self, index):
        original_image_index = self._get_original_index(index)
        x, y = self._get_patch_coods(index)
        cache_path = self.get_cache_path(original_image_index, x, y)
        if os.path.exists(cache_path):
            patch_image = Image.open(cache_path)
        else:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            patch_image = self._get_patch_image(original_image_index, x, y)
            patch_image.save(cache_path)

        # patch_image = self.transforms(patch_image)
        return patch_image

    def __len__(self):
        return self._n_patches

    @property
    def labels(self):
        if len(self._labels) != self.__len__():
            for i in range(self.__len__()):
                original_image_index = self._get_original_index(i)
                _label = self.dataframe["label"][original_image_index]
                _label = int(_label)
                self._labels.append(_label)
        return self._labels

    @property
    def classes(self):
        if len(self._classes) != self.__len__():
            for i in range(self.__len__()):
                original_image_index = self._get_original_index(i)
                _class = self.dataframe["class"][original_image_index]
                self._classes.append(_class)
        return self._classes


class RandomSamplePatchDataset(Dataset):
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        oii = self.dataset._get_original_index(index)
        oi = self.dataset._get_original_image(oii)

        mx, my = oi.size
        x = random.randint(0, mx-225)
        y = random.randint(0, my-255)
        patch = oi.crop((x, y, x+224, y+224))
        patch = self.dataset.transforms(patch)

        label = self.dataset.dataframe["label"][oii]
        return patch, label


class LabellablePatchDataset(Dataset):
    def __init__(self, dataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.labels = [0]*len(self.dataset)

    def set_labels(self, labels):
        self.labels = labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        return x, self.labels[index]


class SpoiledPatchDataset(Dataset):
    def __init__(self, dataset: PatchDataset, label_spoil):
        self.dataset = dataset
        self.indexes = []
        for i in range(len(self.dataset.labels)):
            if self.dataset.labels[i] == label_spoil:
                self.indexes.append(i)

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, index):
        return self.dataset[self.indexes[index]]


class PatchFrom2ImagesDataset(Dataset):
    def __init__(self, dataset: PatchDataset, dummy=False) -> None:
        self.dataset = dataset
        self.ca_dataset = SpoiledPatchDataset(self.dataset, True)
        self.ng_dataset = SpoiledPatchDataset(self.dataset, False)
        self.dummy = dummy

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x1, y = self.dataset[index]

        if y:
            _index = np.random.randint(0, len(self.ca_dataset))
            x2, _ = self.ca_dataset[_index]
        else:
            _index = np.random.randint(0, len(self.ng_dataset))
            x2, _ = self.ng_dataset[_index]
        if self.dummy:
            return (x1, x1), y
        else:
            return (x1, x2), y

    @property
    def labels(self):
        return self.dataset.labels


class OriginalImageDataset(Dataset):
    def __init__(self, dataframe, transforms):
        self.dataframe = dataframe
        self.transforms = transforms

    def __getitem__(self, index):
        image = Image.open(self.dataframe["fullpath"][index])
        label = self.dataframe["label"][index]
        if self.transforms is not None:
            image = self.transforms(image)
        return image, label

    @property
    def labels(self):
        return self.dataframe["label"].values.astype("int")

    def __len__(self):
        return len(self.dataframe)


class ResizedDataset(Dataset):
    def __init__(self, dataframe, transforms, resize=224):
        self.original_dataset = OriginalImageDataset(dataframe, None)
        self.transforms = transforms
        self.resize = resize

    def get_cache_path(self, index):
        basename = self.original_dataset.dataframe["basename"][index]
        return f"./dataset/cache/resized/{self.resize}/{basename}.png"

    def __getitem__(self, index):
        cache_path = self.get_cache_path(index)
        try:
            image = Image.open(cache_path)
        except:
            image, _ = self.original_dataset[index]
            image = image.resize((self.resize, self.resize))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            image.save(cache_path)
        label = self.original_dataset.dataframe["label"][index]

        if self.transforms is not None:
            image = self.transforms(image)
        return image, int(label)

    def __len__(self):
        return len(self.original_dataset)


class ThumbnailDataset(Dataset):
    def __init__(self, dataframe, p=10):
        self.original_dataset = OriginalImageDataset(dataframe, None)
        self.p = p

    def get_cache_path(self, index):
        basename = self.original_dataset.dataframe["basename"][index]
        return f"./dataset/cache/thumbnail/{self.p}/{basename}.png"

    def __getitem__(self, index):
        cache_path = self.get_cache_path(index)
        try:
            image = Image.open(cache_path)

        except:
            image, _ = self.original_dataset[index]
            size = image.size
            image = image.resize((size[0]//self.p, size[1]//self.p))
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            image.save(cache_path)
        label = self.original_dataset.dataframe["label"][index]

        return image, int(label)

    def __len__(self):
        return len(self.original_dataset)


class NikonFileInfoManager:
    def __init__(self, root, index_xlsx, exts=["jpg"]):
        files = self._find_files(root, exts)
        table = self._table_from_list(files)
        sheet = self._read_xlsx(index_xlsx)
        self.table = self._update_table_with_sheet(table, sheet)

    def _find_files(self, root, exts):
        """
            rootディレクトリ以下のすべてのextファイルのリストを作成.
            ファイル名が重複するものは片方を削除.
        """
        files = []
        for ext in exts:
            files.extend(glob.glob(f"{root}**/*.{ext}", recursive=True))

        # 重複削除
        no_duplicated_files = {}
        for _file in files:
            no_duplicated_files[os.path.basename(_file)] = _file
        no_duplicated_files = list(no_duplicated_files.values())
        return no_duplicated_files

    def _table_from_list(self, files):
        """
            ファイル名のリストからDataFrameを作成
        """
        rows = []
        for file in files:
            base = os.path.basename(file)
            row = {}
            row["fullpath"] = file
            row["basename"] = base
            if re.match(r'cell\.[a-z1-9]+?\.[0-9]+?\.[0-9a-z]+?\.[a-zA-Z]+?', base) is not None:
                row["type"] = base.split(".")[0]
                row["class"] = base.split(".")[1]
                row["number"] = base.split(".")[2]
                row["magnification"] = base.split(".")[3]
            elif re.match(r'fna\.cell\.[a-z1-9]+?\.[0-9]+?\.[0-9xX]+?\.[a-zA-Z]+?', base) is not None:
                row["type"] = "fna_cell"
                row["class"] = base.split(".")[2]
                row["number"] = base.split(".")[3]
                row["magnification"] = base.split(".")[4]
            elif re.match(r'fna\.cell\.[a-z1-9]+?\.[0-9]+?\.[a-zA-Z0-9]+?\.[0-9xX]+?\.[a-zA-Z]+?', base) is not None:
                row["type"] = "fna_cell"
                row["class"] = base.split(".")[2]
                row["number"] = base.split(".")[3]
                row["magnification"] = base.split(".")[5]
            elif re.match(r'fna\.cell\.[a-z1-9]+?\.[0-9]+?\.[a-zA-Z0-9]+?\.[0-9]\.[0-9xX]+?\.[a-zA-Z]+?', base) is not None:
                row["type"] = "fna_cell"
                row["class"] = base.split(".")[2]
                row["number"] = base.split(".")[3]
                row["magnification"] = base.split(".")[6]
            rows.append(row)
        return pd.DataFrame(rows)

    def _read_xlsx(self, path):
        sheet = pd.read_excel(path, sheet_name=0, usecols=["検体番号", "永久ＨＥnear1(癌1,正常0）", "永久ＨＥnr2(癌1,正常0）"])
        sheet = sheet.rename(columns={"検体番号": "number", "永久ＨＥnear1(癌1,正常0）": "nr1", "永久ＨＥnr2(癌1,正常0）": "nr2"})
        sheet = sheet.dropna(axis=0)
        sheet = sheet.astype({"number": str, "nr1": bool, "nr2": bool})
        return sheet

    def _update_table_with_sheet(self, table, sheet):
        table["permaHE"] = np.nan
        for _, line in sheet.iterrows():
            table.loc[(table["class"] == "nr1") & (table["number"] == line.number), "permaHE"] = line.nr1
            table.loc[(table["class"] == "nr2") & (table["number"] == line.number), "permaHE"] = line.nr2
        table["label"] = np.nan
        table.loc[table["class"].isin(["ca", "cang"]), "label"] = True
        table.loc[table["class"].isin(["ng"]), "label"] = False
        table.loc[table["class"].isin(["fa", "adh", "idp", "ph"]), "label"] = False

        table.loc[table["class"].isin(["nr1", "nr2"]), "label"] = table["permaHE"]
        return table

    def get_files(self, types, classes, magnifications, labels=[True, False]) -> pd.DataFrame:
        _type = self.table["type"].isin(types)
        _class = self.table["class"].isin(classes)
        _magnification = self.table["magnification"].isin(magnifications)
        _label = self.table["label"].isin(labels)
        query = _type & _class & _magnification & _label
        return self.table[query].reset_index(drop=True)

    def get_info(self, name):
        return self.table[self.table["basename"] == name].iloc[0]
