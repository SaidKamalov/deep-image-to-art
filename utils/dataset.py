import os
import sys

sys.path.append(os.getcwd())

import shutil
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

from api.config import ALLOWED_EXTENSIONS


class CustomDataModule(pl.LightningDataModule):
    """Custom implementation of LightningDataModule."""

    def __init__(
        self,
        path_to_dataset,
        height,
        width,
        train_transform=transforms.ToTensor(),
        test_transform=transforms.ToTensor(),
        train_batch_size=1,
        test_batch_size=1,
        shuffle=False,
        val_size=0.2,
    ) -> None:
        """Constructor of CustomDataModule.

        Args:
            path_to_dataset (str): path to dataset folder.
            height (int): height of images.
            width (int): width of images.
            train_transform: transformation to apply for images in train dataset.
            test_transform: transformation to apply for images in test dataset.
            train_batch_size (int): batch size of train dataset.
            test_batch_size (int): batch size of test dataset.
            shuffle (bool): flag to shuffle train dataset.
            val_size (float): size of validation part in train dataset.
        """
        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.height = height
        self.width = width
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.val_size = val_size

    def prepare_data(self):
        trainA_path = os.path.join(self.path_to_dataset, "trainA/")
        testA_path = os.path.join(self.path_to_dataset, "testA/")
        trainB_path = os.path.join(self.path_to_dataset, "trainB/")
        testB_path = os.path.join(self.path_to_dataset, "testB/")

        self.train_dataset = CustomDataset(
            trainA_path,
            trainB_path,
            self.height,
            self.width,
            self.train_transform,
        )

        self.test_dataset = CustomDataset(
            testA_path,
            testB_path,
            self.height,
            self.width,
            self.test_transform,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train, self.val = random_split(
                self.train_dataset, [1 - self.val_size, self.val_size]
            )

        if stage == "test" or stage is None:
            self.test = self.test_dataset
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train, batch_size=self.train_batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.train_batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.test_batch_size)


class CustomDataset(Dataset):
    """Custom implementtion of pytorch dataset."""

    def __init__(self, path_to_set_A, path_to_set_B, height, width, transform) -> None:
        """Constructor of CustomDataset.

        Args:
            path_to_set_A (str): path to directory which contains images of set A.
            path_to_set_B (str): path to directory which contains images of set B.
            height (int): height of images.
            width (int): width of images.
            transform: transformation to apply for images in dataset.
        """
        super().__init__()
        self.files_A = filter_by_extension(get_files(os.path.abspath(path_to_set_A)))
        self.files_B = filter_by_extension(get_files(os.path.abspath(path_to_set_B)))
        self.height = height
        self.width = width
        self.transform = transform

    def __getitem__(self, index):
        img_file_A, img_file_B = self.files_A[index], self.files_B[index]

        img_A = read_image_to_np(img_file_A, self.width, self.height)
        img_B = read_image_to_np(img_file_B, self.width, self.height)

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return (img_A, img_B)

    def __len__(self):
        return len(self.files_A)


def split_and_save(
    path_to_set_A, path_to_set_B, path_to_dataset, test_size=0.2, shuffle=True
):
    """Take images from directories of set A and set B and split them into test and train directories.

    Args:
        path_to_set_A (str): path to directory which contains images of set A.
        path_to_set_B (str): path to directory which contains images of set B.
        path_to_dataset (str): path to dataset directory where to store train and test directories.
        test_size (float): ratio of test dataset.
        shuffle (bool): flag to shuffle files before splitting.
    """
    img_files_set_A = filter_by_extension(get_files(os.path.abspath(path_to_set_A)))
    img_files_set_B = filter_by_extension(get_files(os.path.abspath(path_to_set_B)))

    length = min(len(img_files_set_A), len(img_files_set_B))
    img_files_set_A = img_files_set_A[:length]
    img_files_set_B = img_files_set_B[:length]

    trainA, testA = train_test_split(
        img_files_set_A, test_size=test_size, shuffle=shuffle
    )
    trainB, testB = train_test_split(
        img_files_set_B, test_size=test_size, shuffle=shuffle
    )

    trainA_path = os.path.join(path_to_dataset, "trainA/")
    testA_path = os.path.join(path_to_dataset, "testA/")
    trainB_path = os.path.join(path_to_dataset, "trainB/")
    testB_path = os.path.join(path_to_dataset, "testB/")

    os.mkdir(path_to_dataset)
    for path in [trainA_path, testA_path, trainB_path, testB_path]:
        os.mkdir(path)

    for a, b in zip(trainA, trainB):
        shutil.copy(a, trainA_path)
        shutil.copy(b, trainB_path)

    for a, b in zip(testA, testB):
        shutil.copy(a, testA_path)
        shutil.copy(b, testB_path)


def filter_by_extension(files):
    """Filter files that have allowed extensions

    Args:
        files (list[str]): list of file names.

    Returns:
        result (list[str]): filtered list of file names.
    """
    result = list(filter(lambda f: get_extension(f) in ALLOWED_EXTENSIONS, files))
    return result


def get_files(dir):
    """Get file names from directory.

    Args:
        dir (str): path to drectory.

    Returns:
        files (list[str]): list of file names.
    """
    files = [
        os.path.join(dir, f)
        for f in os.listdir(dir)
        if os.path.isfile(os.path.join(dir, f))
    ]
    return files


def read_image_to_np(image_file, width, height, resizing_method=Image.BILINEAR):
    """Read image and return as numpy array

    Args:
        image_file (str): path to image.
        width (int): width of an image.
        height (int): height of an image.
        resizing_method: resizing method from PIL.Image.
    """
    img = (
        Image.open(image_file)
        .convert("RGB")
        .resize([width, height], resample=resizing_method)
    )
    img = np.array(img)
    return img


def get_extension(file):
    """Get extensiom of a file.

    Args:
        file (str): name of a file.

    Returns:
        _(str): file extension.
    """
    return os.path.splitext(file)[1][1:]
