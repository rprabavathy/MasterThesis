import torch
import numpy as np
from PIL import Image
from src.datamodule.FullPredictDS import PatchPredictionDataset
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule

from src import logger
log = logger.get_logger(__name__)

class PatchPredictionDataModule(LightningDataModule):
    """LightningDataModule to handle patch extraction and DataLoader for prediction."""
    
    def __init__(self, image_dir, batch_size, num_workers):
        super().__init__()
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        log.info(f"Loading images from {image_dir} and num_workers {num_workers}")
        
    def setup(self, stage=None):
        """Initialize the dataset."""
        log.info(f'Inside setup {self.image_dir}')
        self.dataset = PatchPredictionDataset(self.image_dir)
        
    def predict_dataloader(self):
        """Create DataLoader for batch prediction."""
        log.info(f'Length of dataset : {len(self.dataset)}')
        return DataLoader(self.dataset, batch_size=self.batch_size, num_workers= self.num_workers, pin_memory= True, shuffle=False)

