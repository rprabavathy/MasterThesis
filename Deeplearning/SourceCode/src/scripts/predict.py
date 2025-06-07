import os
import hydra
from omegaconf import DictConfig
from src.callbacks.predictionWriter import CustomWriter
from pytorch_lightning import (
    LightningDataModule,
    LightningModule,
    Trainer,
    )
from src import logger
log = logger.get_logger(__name__)

class Predictor:
    """Class for handling the prediction pipeline."""
    def __init__(self,  model: LightningModule,model_checkpoint,
                 trainer : Trainer, predictData : DictConfig =None):
        if model_checkpoint is not None:
            self.model = model.__class__.load_from_checkpoint(model_checkpoint)
        else:
            self.model = model
        self.model_checkpoint = model_checkpoint
        self.dataset = predictData
        self.trainer = trainer
        
    def process_and_save_predictions(self):
        """Run predictions on all images in the test directory and save the results."""
        log.info(f"Root Image Directory: {self.dataset.image_dir}")
        subdirs = [d for d in os.listdir(self.dataset.image_dir) if os.path.isdir(os.path.join(self.dataset.image_dir, d))]

        for subdir in subdirs:
            subdir_path = os.path.join(self.dataset.image_dir, subdir)
            log.info(f"Processing subdirectory: {subdir}")

            # Initialize the prediction data module
            log.info(f"Instantiating dataModule <{self.dataset}>")
            predictdatamodule: LightningDataModule  = hydra.utils.instantiate(self.dataset, image_dir=subdir_path, _recursive_=False)
            
            log.info(f"Instantiating Trainer predict class :: <{self.trainer}>")
            # trainer: Trainer  = hydra.utils.instantiate(self.trainer, _recursive_=False)
            
            # Run prediction
            pred_list = self.trainer.predict(model=self.model, datamodule=predictdatamodule, ckpt_path=self.model_checkpoint)
            log.info(f"Total prediction batches: {len(pred_list)}")

            # Each batch's predictions are saved by the CustomWriter callback after prediction
            log.info(f"Prediction processing completed for {subdir}. The images are being saved by the CustomWriter.")
