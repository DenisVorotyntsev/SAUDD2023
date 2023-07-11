import os
from PIL import Image

from torch.utils.data import Dataset
import numpy as np
from sklearn.model_selection import KFold
import albumentations as A


def read_image(path: str) -> np.ndarray:
    """
    Read an image from the given path.

    Args:
        path (str): The path to the image.

    Returns:
        np.ndarray: The image as a numpy array.
    """
    image = np.array(Image.open(path))
    return image


def read_depth_image(path: str) -> np.ndarray:
    """
    Read a depth image from the given path.

    Args:
        path (str): The path to the depth image.

    Returns:
        np.ndarray: The depth image as a numpy array.
    """
    arr = read_image(path)
    mask = arr == 0
    f_img = (arr - 1) / 128.0
    f_img[mask] = 0
    return np.float32(f_img)


class DepthDataset(Dataset):
    def __init__(
        self,
        fold_number: int,
        dataset_mode: str,
        transform_func: callable,
        debug: bool = False,
        max_files: int = None,
    ):
        """
        Initialize the DepthDataset.

        Args:
            fold_number (int): The fold number for KFold split. Use -1 for pre-defined split.
            dataset_mode (str): The mode of the dataset, either 'train' or 'test'.
            transform_func (callable): The transform function to be applied to the data.
            debug (bool, optional): Whether to run the dataset in debug mode. Defaults to False.
            max_files (int, optional): The maximum number of files to include in the dataset. Defaults to None.
        """
        self.root_data_folder = "../data"
        self.dataset_mode = dataset_mode
        self.transform_func = transform_func
        self.fold_number = fold_number

        if self.fold_number >= 0:
            print("Using k fold split")
            files = [
                f
                for f in os.listdir(os.path.join(self.root_data_folder, "inputs"))
                if f.endswith(".png")
            ]
            all_series = [f.split("-")[0] for f in files]
            all_series = list(set(all_series))
            all_series = list(sorted(all_series))
            folds = list(
                KFold(n_splits=5, random_state=42, shuffle=True).split(
                    all_series, all_series
                )
            )
            ind_train_series = folds[fold_number][0]
            ind_test_series = folds[fold_number][1]
            train_series = list(np.array(all_series)[ind_train_series])
            test_series = list(np.array(all_series)[ind_test_series])
            train_files = [
                f for f in files if f.split("-")[0] in train_series
            ]  # file name: SSSSS-TTTTT; SSSSS - flight id, TTTTT - pic timestamp
            test_files = [f for f in files if f.split("-")[0] in test_series]
            print(f"Train series: {len(train_series)}")
            print(f"Test series: {len(test_series)}")
            print(train_series[:3])
            print(test_series[:3])
            print(f"Train files: {len(train_files)}")
            print(f"Test files: {len(test_files)}")
            print(train_files[:3])
            print(test_files[:3])
        else:
            print("Using pre-defined split")
            train_files = os.listdir("../data/train/depth_annotations")
            test_files = os.listdir("../data/test/depth_annotations")

        print("train files", len(train_files))
        print("test files", len(test_files))

        if self.dataset_mode == "train":
            self.image_files = train_files
            self.aug_transform = A.Compose(
                [
                    A.OneOf(
                        [
                            A.HorizontalFlip(),
                            A.VerticalFlip(),
                        ],
                        p=1.0,
                    ),
                ],
                p=0.5,
            )
        else:
            self.image_files = test_files
            self.aug_transform = None

        if debug:
            self.image_files = self.image_files[:8]

        if max_files is not None:
            self.image_files = self.image_files[:max_files]

        print("\n\n")

    def _get_path(self, f: str) -> tuple:
        """
        Get the paths for the image and mask.

        Args:
            f (str): The file name.

        Returns:
            tuple: The paths for the image and mask.
        """
        inp_path = os.path.join(self.root_data_folder, "inputs", f)
        if self.fold_number >= 0:
            mask_path = os.path.join(self.root_data_folder, "depth_annotations", f)
        else:
            mask_path = os.path.join(
                self.root_data_folder, self.dataset_mode, "depth_annotations", f
            )
        return inp_path, mask_path

    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset.

        Args:
            index (int): The index of the item.

        Returns:
            tuple: The image and depth.
        """
        inp_path, mask_path = self._get_path(self.image_files[index])

        img = read_image(inp_path) / 255.0
        img = np.stack((img,) * 3, axis=-1)
        depth = read_depth_image(mask_path)

        if self.aug_transform is not None:
            res = self.aug_transform(image=img, mask=depth)
            img = res["image"]
            depth = res["mask"]

        t_image = self.transform_func({"image": img, "depth": depth, "mask": depth})
        return t_image["image"], t_image["depth"]

    def __len__(self) -> int:
        """
        Get the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.image_files)
