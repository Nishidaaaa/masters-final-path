import pytorch_lightning as pl
import torchvision.transforms as transforms
import torch
import src
from src.utils.data import PathMNISTInfoManager, PatchDataset, PatchFrom2ImagesDataset, RandomSamplePatchDataset, ResizedDataset
import mlflow
import os
from torch.utils.data import DataLoader, Subset

flip_augmentations = transforms.Compose(
    [
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip()
    ]
)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
totensor_normalize = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


def split_files(files):
    n_files = len(files)

    n_train = int(n_files*0.5)
    n_val = int(n_files*0.2)
    n_test = int(n_files*0.3) #あやしい

    train_files = files[0:n_train].reset_index()
    val_files = files[n_train:n_train+n_val].reset_index()
    test_files = files[n_train+n_val:].reset_index()
    return {"train": train_files, "val": val_files, "test": test_files}


class ResizeDataModule(pl.LightningDataModule):
    def __init__(self, hparams) -> None:
        self.save_hyperparameters(hparams, logger=False)

        super().__init__()

    def setup(self, stage: str = "") -> None:
        self.augmentation = flip_augmentations
        self.preprocess = totensor_normalize
        manager = PathMNISTInfoManager(**self.hparams.filemanager)
        files = manager.get_files(**self.hparams.file_spoiler)
        files = files.sample(frac=1, random_state=self.hparams.split_random_state, ignore_index=True)
        self.files = split_files(files)

        for phase in src.PHASES:
            path = f"./tmp/{phase}.csv"
            self.files[phase].to_csv(path)
            mlflow.log_artifact(local_path=path)

        self.resized_datasets = {}
        for phase in src.PHASES:
            _transforms = self.preprocess if phase == "test" else transforms.Compose([self.augmentation, self.preprocess])
            self.resized_datasets[phase] = ResizedDataset(dataframe=self.files[phase], resize=self.hparams.image_size, transforms=_transforms)

    def dataloader(self, phase, shuffle) -> DataLoader:
        return DataLoader(self.resized_datasets[phase], batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        return self.dataloader("train", shuffle)

    def val_dataloader(self, shuffle=False) -> DataLoader:
        return self.dataloader("val", shuffle)

    def test_dataloader(self, shuffle=False) -> DataLoader:
        return self.dataloader("test", shuffle)


class PatchDataModule(pl.LightningDataModule):
    def __init__(self, hparams) -> None:
        self.save_hyperparameters(hparams, logger=False)
        super().__init__()

    def setup(self, stage: str = "") -> None:
        self.augmentation = flip_augmentations
        self.preprocess = totensor_normalize
        manager = PathMNISTInfoManager(**self.hparams.filemanager)
        files = manager.get_files(**self.hparams.file_spoiler)
        files = files.sample(frac=1, random_state=self.hparams.split_random_state, ignore_index=True)
        self.files = split_files(files)

        for phase in src.PHASES:
            path = f"./tmp/{phase}.csv"
            os.makedirs("./tmp", exist_ok=True)
            self.files[phase].to_csv(path)
            mlflow.log_artifact(local_path=path)

        self.patch_datasets = {}
        for phase in src.PHASES:
            _transforms = self.preprocess if phase == "test" else transforms.Compose([self.augmentation, self.preprocess])
            self.patch_datasets[phase] = PatchDataset(dataframe=self.files[phase], patch_size=self.hparams.image_size, transforms=_transforms)

    def dataloader(self, phase, shuffle) -> DataLoader:
        return DataLoader(self.patch_datasets[phase], batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers)

    def train_dataloader(self, shuffle=True, original=False) -> DataLoader:
        return self.dataloader("train", shuffle)

    def val_dataloader(self, shuffle=False, original=False) -> DataLoader:
        return self.dataloader("val", shuffle)

    def test_dataloader(self, shuffle=False, original=False) -> DataLoader:
        return self.dataloader("test", shuffle)

    @property
    def cumulative_sums(self):
        return {phase: self.patch_datasets[phase].cumulative_sum for phase in src.PHASES}

    @property
    def patch_labels(self):
        return {phase: self.patch_datasets[phase].labels for phase in src.PHASES}

    @property
    def original_labels(self):
        return {phase: self.patch_datasets[phase].original_labels for phase in src.PHASES}

    @property
    def patch_classes(self):
        return {phase: self.patch_datasets[phase].classes for phase in src.PHASES}

    @property
    def original_classes(self):
        return {phase: self.patch_datasets[phase].original_classes for phase in src.PHASES}

    @property
    def original_fullpaths(self):
        return {phase: self.patch_datasets[phase].original_fullpaths for phase in src.PHASES}


class PatchFrom2ImagesDataModule(PatchDataModule):
    def setup(self, stage: str = None) -> None:
        super().setup(stage)
        from2datasets = {}
        from2datasets["train"] = PatchFrom2ImagesDataset(self.patch_datasets["train"], False)
        from2datasets["val"] = PatchFrom2ImagesDataset(self.patch_datasets["val"], True)
        from2datasets["test"] = PatchFrom2ImagesDataset(self.patch_datasets["test"], True)

    def train_dataloader(self, shuffle=False) -> DataLoader:
        return super().train_dataloader(shuffle)


class PatchRandomSampleDataModule(PatchDataModule):
    def setup(self, stage: str = None) -> None:
        super().setup(stage)
        self.train_dataset = RandomSamplePatchDataset(self.patch_datasets["train"])

    def train_dataloader(self, shuffle=False, original=False) -> DataLoader:
        if original:
            return super().train_dataloader(shuffle)
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers)