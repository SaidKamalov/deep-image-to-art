import os
import sys
import argparse
from pathlib import Path
from torchvision import transforms

sys.path.append(os.getcwd())

from utils.dataset import CustomDataModule, split_and_save

PROJECT_DIR = "/".join(__file__.split("/")[:-2])
PATH_TO_DATASETS = os.path.join(PROJECT_DIR, "datasets/")

DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--dataset_name",
    type=str,
    required=True,
    help="Name of the dataset you want to have",
)
argparser.add_argument(
    "--path_to_set_A",
    type=str,
    required=False,
    help="Path to the input images of set A",
)
argparser.add_argument(
    "--path_to_set_B",
    type=str,
    required=False,
    help="Path to the input images of set B",
)
argparser.add_argument(
    "--height",
    type=int,
    required=False,
    default=128,
    help="height of images",
)
argparser.add_argument(
    "--width",
    type=int,
    required=False,
    default=128,
    help="width of images",
)
argparser.add_argument(
    "--train_batch_size",
    type=int,
    required=False,
    default=16,
    help="batch size for train dataloader",
)
argparser.add_argument(
    "--test_batch_size",
    type=int,
    required=False,
    default=1,
    help="batch size for test dataloader",
)
argparser.add_argument(
    "--shuffle_dataset",
    type=bool,
    required=False,
    default=False,
    help="shuffle files before splitting",
)
argparser.add_argument(
    "--shuffle_train",
    type=bool,
    required=False,
    default=True,
    help="shuffle dataloader",
)
argparser.add_argument(
    "--val_size",
    type=float,
    required=False,
    default=0.1,
    help="size of validation part",
)
argparser.add_argument(
    "--test_size",
    type=float,
    required=False,
    default=0.2,
    help="size of validation part",
)


def prepare_dataset(
        dataset_name, path_to_set_A=None, path_to_set_B=None, shuffle=False, test_size=0.2, **kwargs
):
    """Create CustomDataModule and save images properly in specific dataset directory if necessary.

    Args:
        dataset_name: name of dataset.
        path_to_set_A (str): path to directory which contains images of set A.
        path_to_set_B (str): path to directory which contains images of set B.
        shuffle (bool): flag to shuffle files before splitting.
        test_size (float): ratio of test dataset.
        kwargs: other parameters for CustomDataModule.

    Returns:
        data_module (CustomDataModule)
    """
    dataset_path = os.path.join(PATH_TO_DATASETS, dataset_name)

    if not is_dataset_present(dataset_path):
        assert (
            path_to_set_A and path_to_set_B
        ), "dataset does not exist, paths to set A and set B are necessary."

        split_and_save(path_to_set_A, path_to_set_B, dataset_path, test_size, shuffle)

    height = kwargs.get("height", 128)
    width = kwargs.get("width", 128)
    train_batch_size = kwargs.get("train_batch_size", 16)
    test_batch_size = kwargs.get("test_batch_size", 1)
    shuffle_dm = kwargs.get("shuffle_train", True)
    val_size = kwargs.get("val_size", 0.1)
    train_transform = kwargs.get("train_transform", DEFAULT_TRANSFORM)
    test_transform = kwargs.get("train_transform", DEFAULT_TRANSFORM)

    data_module = CustomDataModule(
        path_to_dataset=dataset_path,
        height=height,
        width=width,
        train_transform=train_transform,
        test_transform=test_transform,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        shuffle=shuffle_dm,
        val_size=val_size,
    )
    return data_module


def is_dataset_present(dataset_path):
    """Check if directory of dataset exists and contains necessary train and test directories.

    Args:
        dataset_path (str): path to dataset directory

    Returns:
        result (bool): verdict, true if present.
    """
    result = False
    if os.path.exists(dataset_path):
        subdirs = [
            os.path.basename(os.path.normpath(str(path)))
            for path in Path(dataset_path).glob("*/")
        ]
        for p in ["testA", "trainB", "testB", "trainA"]:
            assert p in subdirs, "Invalid dataset structure."
        result = True
    return result


if __name__ == "__main__":
    args = argparser.parse_args()
    kwargs = {
        "height": args.height,
        "width": args.width,
        "train_batch_size": args.train_batch_size,
        "test_batch_size": args.test_batch_size,
        "shuffle_train": args.shuffle_train,
        "val_size": args.val_size,
    }
    dm = prepare_dataset(
        dataset_name=args.dataset_name,
        path_to_set_A=args.path_to_set_A,
        path_to_set_B=args.path_to_set_B,
        shuffle=args.shuffle_dataset,
        test_size=args.test_size,
        kwargs=kwargs,
    )
