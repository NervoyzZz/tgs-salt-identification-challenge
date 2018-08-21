"""Module defines functions to manipulate with data."""

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset


class TGSDataset(Dataset):
    """
    Class to create torch.utils.data.Dataset
    """

    def __init__(self, data_path, image_key, mask_key, name_key, sample_count):
        """
        Create dataset

        Args:
            data_path: str
                Path to file with data
            image_key: tr
                Key for extracting input images
            mask_key: str
                Key for extracting output masks
            name_key: str
                Key for extracting names of images
            sample_count: int
                How much samples should be load
                -1 for whole data

        """
        dataset = None # read data
        img = np.array(dataset[image_key][:sample_count])
        msk = np.array(dataset[mask_key][:sample_count])
        nms = np.array(dataset[name_key][:sample_count])
        img = img.reshape(img.shape[0], -1, img.shape[1], img.shape[2])
        msk = msk.reshape(msk.shape[0], -1, msk.shape[1], msk.shape[2])
        nms = nms.reshape(nms.shape[0], -1, nms.shape[1], nms.shape[2])
        self.images = torch.from_numpy(img)
        self.masks = torch.from_numpy(msk)
        self.names = nms

    def __getitem__(self, index):
        """
        Get set of Image[index], Mask[index], Name[index]

        Args:
            index: int
                Index to find a set

        Returns: numpy.array, numpy.array, numpy.array:
            Image, mask and name for this image

        """
        return self.images[index], self.masks[index]

    def __len__(self):
        """
        Get number of images in dataset

        Returns:
            int:
                Number of images in dataset

        """
        return len(self.images)


def create_data_loader(dataset, batch_size, cpu_count=1):
    """
    Create DataLoader from dataset

    Args:
        dataset: TGSDataset
            Dataset that should be loaded
        batch_size: int
            Batch size for loader
        cpu_count: int
            Count of cpu on PC to make loading faster

    Returns:
        torch.utils.data.DataLoader:
            Dataloader for dataset

    """
    loader = DataLoader(dataset, batch_size=batch_size,
                        num_workers=2 * cpu_count)
    return loader
