import torch
import numpy as np
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Subset
from src.datamodule.RandomDataset import BrainSectionDataset
from src import logger

logger = logger.get_logger(__name__)

class BrainSectionDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        batch_size: int = 32, 
        num_workers: int = 4, 
        pin_memory: bool = True, 
        dataset : DictConfig =None,
        transform: DictConfig =None
        ):
        super().__init__()
        
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.save_hyperparameters(logger=False)
        logger.info(f'initialization : {dataset}')
        self.dataset_cfg = dataset
        self.transform = transform
        
        self.dataset = None
        
    def prepare_data(self):
        """Runs only ONCE. Load dataset without applying splits."""
        if self.dataset is None:
            logger.info("Preparing dataset (only runs once in distributed training)")
            self.dataset = BrainSectionDataset(
                image_dir=self.dataset_cfg.image_dir,
                mask_dir=self.dataset_cfg.mask_dir,
                patch_size=self.dataset_cfg.patch_size,
                num_patches=self.dataset_cfg.num_patches,
                transform=None  # No transforms applied here
            )
        
    def setup(self, stage=None):
        num_gpus = torch.cuda.device_count()  # Get available GPUs
        if num_gpus > 1:  
            self.hparams.batch_size = self.hparams.batch_size // num_gpus  # Adjust batch size for DDP

        logger.info(f"Using {num_gpus} GPUs -> Adjusted batch size: {self.hparams.batch_size}")
        
        logger.debug(f"Setup called with stage: {stage}")
        if self.dataset is None:
            self.prepare_data()  # Ensure dataset is loaded

        train_idx, val_idx = self.dataset.getSplit(self.dataset_cfg.val_size, self.dataset_cfg.test_size  )
        logger.info(f"Dataset split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        if stage == 'fit' or stage is None:
            self.train_dataset = Subset(self.dataset, train_idx)
            self.train_dataset.dataset.transform =  self.transform
            logger.info(f"Training dataset length: {len(self.train_dataset)}")
            self.val_dataset = Subset(self.dataset, val_idx)
            self.val_dataset.dataset.transform = None
            logger.info(f"Validation dataset length: {len(self.val_dataset)}")
        
 #       if stage == 'test' or stage is None:
 #           self.test_dataset = Subset(self.dataset, test_idx)
 #           self.test_dataset.dataset.transform = None
 #           logger.info(f"Test dataset length: {len(self.test_dataset)}")

    def train_dataloader(self):
       return DataLoader(
           self.train_dataset,
           batch_size=self.hparams.batch_size,
           num_workers=self.hparams.num_workers,
           pin_memory=self.hparams.pin_memory,
          shuffle=True,
       )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

#    def test_dataloader(self):
#        return DataLoader(
#            self.test_dataset,
#            batch_size=self.hparams.batch_size,
#            num_workers=self.hparams.num_workers,
#            pin_memory=self.hparams.pin_memory,
#            shuffle=False,
#            )

