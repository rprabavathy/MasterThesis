import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from src import logger
log = logger.get_logger(__name__)

class PatchPredictionDataset(Dataset):
    """Dataset that loads full images from a folder."""
    def __init__(self, image_dir, pad_to_multiple=32):
        self.image_dir = image_dir
        self.image_paths = []
        self.pad_to_multiple = pad_to_multiple
        
        # Traverse the directory and collect image paths
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):  # Add other extensions as needed
                    self.image_paths.append(os.path.join(root, file))
        
        log.info(f"Found {len(self.image_paths)} images in {image_dir}")

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an image and return it as a tensor."""
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert('RGB'))
        # Get the original dimensions
        height, width, _ = image.shape
        log.debug(f'Image Size : {image.shape}')
        # Compute how much padding is required to make both dimensions divisible by pad_to_multiple (e.g., 32)
        pad_h = (self.pad_to_multiple - height % self.pad_to_multiple) % self.pad_to_multiple
        pad_w = (self.pad_to_multiple - width % self.pad_to_multiple) % self.pad_to_multiple

        # Store the padding values for unpadding later
        padding_info = {"pad_h": pad_h, "pad_w": pad_w}

        # Pad the image if necessary
        padded_image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

        # Convert the image to tensor and permute the dimensions to (C, H, W)
        image_tensor = torch.tensor(padded_image).permute(2, 0, 1).float()
        
        return image_tensor, padding_info
