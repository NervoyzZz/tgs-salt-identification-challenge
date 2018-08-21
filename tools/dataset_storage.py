# Basic imports
import os
import numpy as np
import cv2
import torch


class DatasetStorage:
    """Class for saving and loading TGS images and masks"""

    def __init__(self):
        """Initialize empty dataset storage."""

        self.filenames = []
        self.images = []
        self.masks = None

    def load_dirs(self, images_dir, masks_dir=None):
        """Load images from directories.

        Parameters
        ----------
        images_dir : str
            Path to the images (train or test)
        masks_dir : str
            Path to the masks (train) or None
        """

        self.filenames = []
        self.images = []
        if masks_dir is not None:
            self.masks = []
        else:
            self.masks = None

        for file in os.listdir(images_dir):
            self.filenames.append(file)

            image_path = os.path.join(images_dir, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.images.append(image)

            if masks_dir is not None:
                mask_path = os.path.join(masks_dir, file)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                self.masks.append(mask)

    def load_aggregated(self, aggregated_path):
        """Load images from aggregated file.

        Parameters
        ----------
        aggregated_path : str
            Path to the aggregated file with images
            and (optionally) masks
        """

        aggregated = torch.load(aggregated_path)
        self.filenames = aggregated["filenames"]
        self.images = aggregated["images"]
        self.masks = aggregated["masks"]

    def save_aggregated(self, aggregated_path):
        """Save images to aggregated file.

        Parameters
        ----------
        aggregated_path : str
            Path to the aggregated file with images
            and (optionally) masks
        """

        aggregated = {
            "filenames": self.filenames,
            "images": self.images,
            "masks": self.masks
        }
        torch.save(aggregated, aggregated_path)

    def __getitem__(self, item):
        if self.masks is None:
            return self.images[item]
        else:
            return self.images[item], self.masks[item]

    def __len__(self):
        return len(self.images)
