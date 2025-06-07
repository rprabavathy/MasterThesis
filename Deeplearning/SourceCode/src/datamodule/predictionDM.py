import numpy as np
from PIL import Image
from src.datamodule.predictionDS import PatchPredictionDataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src import logger
log = logger.get_logger(__name__)

class PatchPredictionDataModule(LightningDataModule):
    """LightningDataModule to handle patch extraction and DataLoader for prediction."""
    def __init__(self, image_path, patch_size, spacing, batch_size):
        super().__init__()
        self.image_path = image_path
        self.patch_size = (patch_size,patch_size)
        self.spacing = spacing
        self.batch_size = batch_size
        log.debug(f'processing image {image_path}')
        log.info(f' patch size : {patch_size} and spacing: {spacing} and batch size: {batch_size}')

    def setup(self, stage=None):
        """Setup the dataset for prediction."""
        # Load the image and convert it to a NumPy array (RGB)
        image = np.array(Image.open(self.image_path).convert('RGB'))
        # Initialize the dataset to extract patches from the image
        self.dataset = PatchPredictionDataset(image, self.patch_size, self.spacing)

    def predict_dataloader(self):
        """Create the DataLoader for prediction."""
        log.info(f'prediction dataset length per image : {len(self.dataset)}')
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def teardown(self, stage=None):
        """Clean up any resources, if necessary."""
        self.image = None
