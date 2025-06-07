import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

from src import logger
log = logger.get_logger(__name__)

class PatchPredictionDataset(Dataset):
    """Custom Dataset for patch-based prediction with padding and reconstruction methods."""

    def __init__(self, image, patch_size, spacing):
        """
        Args:
            image (numpy.array): The input image for patch extraction.
            patch_size (tuple): The size of each patch (height, width).
            spacing (int): The spacing between patches, which determines the overlap.
        """
        self.image = image
        self.patch_size = patch_size
        self.spacing = spacing
        
        # Apply padding to image to make sure dimensions are divisible by patch size
        self.padded_image, self.pad_value = self.apply_padding(image, patch_size)
        
        # Generate grid and coordinates for patch extraction
        self.coords = self.create_grid(self.padded_image, self.spacing)
        
        # Extract patches from the image
        self.patches = self.sample_from_coords(self.padded_image, self.coords, self.patch_size)
        
        # Ensure patches are in a format suitable for the model (C, H, W)
        self.patches = [torch.tensor(np.array(patch)).permute(2, 0, 1).float() for patch in self.patches]

    def __len__(self):
        """Returns the number of patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """
        Returns:
            torch.Tensor: A single patch from the dataset.
        """
        return self.patches[idx]            

    # --- Methods for padding, patch extraction, and image reconstruction ---

    def apply_padding(self, img, patch_size):
        """ Apply padding to make sure the image dimensions are divisible by patch size """
        # Calculate padding required for height and width
        height, width = img.shape[:2]
        pad_height = (patch_size[0] - height % patch_size[0]) % patch_size[0]
        pad_width = (patch_size[1] - width % patch_size[1]) % patch_size[1]
        
        # Apply the padding uniformly on both sides
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left
        
        # Perform the padding
        if len(img.shape) == 3:  # RGB image (3 channels)
            padded_img = np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='reflect')
        else:  # Grayscale image (1 channel)
            padded_img = np.pad(img, ((top, bottom), (left, right)), mode='reflect')
        
        pad_value = (top, left, bottom, right)
        return padded_img, pad_value

    def remove_padding(self, padded_img, pad_value):
        """ Remove padding from the padded image """
        # Slice the padded image to remove padding
        original_img = padded_img[pad_value[0]:padded_img.shape[0] - pad_value[2], 
                                  pad_value[1]:padded_img.shape[1] - pad_value[3]]
        
        return original_img

    def create_grid(self, img, spacing):
        """ Create grid of coordinates for patch extraction """
        xx, yy = np.meshgrid(range(spacing//2, img.shape[0], spacing), 
                             range(spacing//2, img.shape[1], spacing))
        return np.array((xx.ravel(), yy.ravel())).T

    def sample_from_coords(self, img, coords, patch_shape):
        """ Sample patches from the image at the specified coordinates """
        crops = []
        offset = np.array(patch_shape)//2
        for coord in coords:
            coord = np.array(coord)  # coord is center point of crop, coord-offset is upper left corner of crop in img
            ul = coord - offset  # coord-offset is upper left corner in img
            lr = ul + np.array(patch_shape)
            crop = img[ul[0]:lr[0], ul[1]:lr[1]]
            crops.append(crop.astype(np.uint8))
        return crops

    def create_image_from_patches(self, patches, coords, shape):
        """ Reconstruct the image from patches by placing them in their respective locations """
        
        log.info(f'Patches Shape {patches.shape}')
        # Ensure patches are 4D (N, patch_height, patch_width, 3)
        if len(patches.shape) == 3:  # If patches are 3D (N, height, width), add channel dimension
            patches = patches[:, :, :, np.newaxis]

        image = np.zeros(shape, dtype=np.uint8)  # Initialize the image with zeros
        padded_image, pad_value = self.apply_padding(image, patches.shape[1:3])  # Apply padding to the image
        
        # Initialize result image with RGB shape
        res = np.zeros(padded_image.shape, dtype=np.uint8)  # Shape of the final image (height, width, 3)
        offset = np.array(patches.shape[1:3]) // 2  # Calculate the offset for patch placement
        
        for i, coord in enumerate(coords):
            coord = np.array(coord)
            ul = coord - offset  # Upper-left corner of the patch
            lr = ul + np.array(patches.shape[1:3])  # Lower-right corner of the patch
            lr = np.min([ul + np.array(patches.shape[1:3]), padded_image.shape[:2]], axis=0)  # Lower-right corner
            
            # Slice the patch (clip any part that overflows)
            patch = patches[i, 0:lr[0]-ul[0], 0:lr[1]-ul[1], :]  # Extract RGB channels

            # Place the patch into the result image
            res[ul[0]:lr[0], ul[1]:lr[1], :] = patch
        # Average the overlapping regions by dividing by the count of overlaps
        original_img = self.remove_padding(res, pad_value)
        return original_img
