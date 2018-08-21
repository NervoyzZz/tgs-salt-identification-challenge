# Basic imports
import os
import numpy as np
import cv2
import torch


class DatasetStorage:
    """Class for saving and loading TGS images and masks"""

    def __init__(self, images_dir, masks_dir=None):
        """Initialize dataset storage.

        Parameters
        ----------
        images_dir : str
            Path to the images (train or test)
        masks_dir : str
            Path to the masks (train) or None
        """

        self.filenames = []
        self.images = []
        self.masks = []
        if masks_dir is None:
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

    def __getitem__(self, item):
        if self.masks is None:
            return self.images[item]
        else:
            return self.images[item], self.masks[item]

    def __len__(self):
        return len(self.images)
