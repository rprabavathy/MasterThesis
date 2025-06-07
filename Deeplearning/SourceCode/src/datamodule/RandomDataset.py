import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from src import logger

log = logger.get_logger(__name__)

class BrainSectionDataset(Dataset):
    def __init__(self, image_dir, mask_dir, patch_size=256, num_patches=20000, transform=None):
        
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.transform = transform
        
        log.info(f'Image_dir : {image_dir} and Mask_dir:{mask_dir}')
        image_files, mask_files = sorted(os.listdir(image_dir)), sorted(os.listdir(mask_dir))
        assert len(image_files) == len(mask_files)
        self.data = pd.DataFrame({
            'image': image_files,
            'mask': mask_files
        })
        
        assert len(self.data) > 0, "No images found in the specified directory."
        # Load images and masks only once into memory
        self.images = [self.load_image(os.path.join(image_dir, fname)) for fname in self.data['image']]
        self.masks = [self.load_image(os.path.join(mask_dir, fname), is_mask=True) for fname in self.data['mask']]
        
        self.patch_per_image = (int(self.num_patches / len(self.images)) + 199) // 200 * 200
        log.info(f'patches_per_images : {self.patch_per_image}')
        self.BG_class = []
        self.tissue_class = []
        self.patches = self._create_patches()

    def load_image(self, path, is_mask=False):
        """ Load image or mask from path """
        img = Image.open(path).convert("L" if is_mask else "RGB")
        return np.array(img)
    
    def _create_patches(self):
        """
        Populate the lists for full mask patches and mixed patches across all images.
        """
        for img, mask in zip(self.images, self.masks):
            h, w = img.shape[:2]

            for _ in range(self.patch_per_image):  # Generate num_patches randomly for each image
                # Randomly select a top-left corner (x, y) for the patch
                x = random.randint(0, w - self.patch_size)
                y = random.randint(0, h - self.patch_size)

                img_patch = img[y:y+self.patch_size, x:x+self.patch_size]
                mask_patch = mask[y:y+self.patch_size, x:x+self.patch_size]
                label_map = {0: 0, 100: 1, 200: 2, 255:2}
                for original_label, new_label in label_map.items():
                    mask_patch = np.where(mask_patch == original_label, new_label, mask_patch)
                if np.max(mask_patch) == 0:
                    self.BG_class.append((img_patch, mask_patch))
                else:
                    self.tissue_class.append((img_patch, mask_patch))

        log.info(f"Total Background Patches: {len(self.BG_class)}, Total Tissue Patches: {len(self.tissue_class)}")

    def __len__(self):
        """
        Total patches = number of full mask patches + number of mixed patches
        """
        #return (len(self.tissue_class) + len(self.BG_class))
        return self.num_patches

    def __getitem__(self, idx):
        """
        Randomly select a patch (background or tissue class).
        """
        # Determine whether we are getting a background or tissue class
        if idx < int(self.num_patches * 0.10):  # 15% full mask patches
            img_patch, mask_patch = random.choice(self.BG_class)
        else:  # 85% mixed patches
            img_patch, mask_patch = random.choice(self.tissue_class)

        # Apply any transformations if defined
        if self.transform is not None and np.max(mask_patch)!=0:
            augmented = self.transform(image=img_patch, mask=mask_patch)
            img_patch = augmented['image']
            mask_patch = augmented['mask']

        # Convert to (C, H, W) format for PyTorch
        img_patch = img_patch.transpose(2, 0, 1)
        return img_patch, mask_patch

    def getSplit(self, val_size=0.15, test_size=0.15, random_seed=42):
        """
        Splits dataset indices into train, validation, and test based on patches.
        Returns three lists of indices: train_idx, val_idx, test_idx.
        """
        log.info(f'Patch_size : {self.patch_size}')
        indices = list(range(len(self)))
        labels = np.array([self[i][1].flatten() for i in indices])
        patch_labels = np.array([np.bincount(label).argmax() for label in labels])

        # Log original class counts with percentages
        unique_labels, counts = np.unique(patch_labels, return_counts=True)
        log.info(f'unique labels : {unique_labels} and its count : {counts}')
        total_count = counts.sum()
        class_distribution = {
            int(label): (int(count), f"{(count / total_count) * 100:.2f}%")
            for label, count in zip(unique_labels, counts)
        }
        log.info(f"Original class distribution: {class_distribution}")

        def log_class_distribution(split_name, split_indices):
            split_labels = patch_labels[split_indices]
            unique_split_labels, split_counts = np.unique(split_labels, return_counts=True)
            total_samples = len(split_indices)

            log.info(f"{split_name} class distribution length : {len(split_indices)}")
            for label, count in zip(unique_split_labels, split_counts):
                percentage = (count / total_samples) * 100
                log.info(f"  Class {label}: {count} samples ({percentage:.2f}%)")
        

        train_idx, val_idx = train_test_split(indices,
                                          stratify=patch_labels,
                                          test_size=val_size,
                                          random_state=random_seed)
       # train_idx, temp_idx = train_test_split(indices,
       #                                        stratify=patch_labels,
       #                                        test_size=(test_size / (val_size + test_size)),
       #                                        random_state=random_seed)
       # val_idx, test_idx = train_test_split(temp_idx,
       #                                      stratify=patch_labels[temp_idx],
       #                                      test_size=(test_size / (val_size + test_size)),
       #                                      random_state=random_seed)
        log_class_distribution("Train", train_idx)
        log_class_distribution("Validation", val_idx)
       # log_class_distribution("Test", test_idx)

        return train_idx, val_idx #, test_idx

