import os
import torch
import hydra
from typing import List, Optional
from omegaconf import DictConfig
from src.scripts.predict import Predictor
from pytorch_lightning.loggers import Logger
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
    )

from src import logger
log = logger.get_logger(__name__)

def train(config: DictConfig) -> Optional[float]:
    """Training pipeline.
    Args:
        config (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """
    # Set seed for reproducibility
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    
    log.debug(f'CUDA Availability : {torch.cuda.is_available()}')  # Should return True if GPUs are available
    log.debug(f'Device count : {torch.cuda.device_count()} and Current Device : {torch.cuda.current_device()}')
    #torch.set_float32_matmul_precision('high')

    log.info(f"Instantiating datamodule <{config.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Instantiate model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)
    
    log.info(f"Model device: {next(model.parameters()).device}")

    callbacks: List[Callback] = [] # Instantiate callbacks
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items(): 
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))
    log.debug("Callbacks passed to trainer:", callbacks)
    
    logger: List[Logger] = [] # Instantiate loggers
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    # Instantiate trainer
    log.info(f"Instantiating trainer <{config.trainer._target_}>")
    train_trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    
    log.info(f"Trainer configuration: {config.trainer.__dict__}")
    log.info(f"DDP initialized: {train_trainer.global_rank != -1}")
    log.info(f"Trainer backend: {train_trainer.strategy}")
 
    # Train the model
    log.info("Starting training!")
    train_trainer.fit(model=model, datamodule=datamodule)
    
    # Save the final model checkpoint
    final_checkpoint_path = os.path.join(config.callbacks.model_checkpoint.dirpath, "final_model.ckpt")
    train_trainer.save_checkpoint(final_checkpoint_path)
    log.info(f"Final model saved at {final_checkpoint_path}")
    

    predict_trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )

    log.info(f"Instantiating trainer for validation/testing <{config.trainer._target_}>")
    # Update configuration for single-GPU (for validation/testing phase)
    config.trainer.devices = 1  # Use 1 GPU for validation/testing
    # Remove the strategy key entirely for single-GPU (validation/testing)
    if "strategy" in config.trainer:
        del config.trainer.strategy

    val_trainer: Trainer = hydra.utils.instantiate(
        config.trainer, callbacks=callbacks, logger=logger, _convert_="partial"
    )
    log.info(f"DDP initialized: {val_trainer.global_rank != -1}")
    log.info(f"Val/Test backend: {val_trainer.strategy}")

    # Validate the model
    log.info("Validating the model...")
    val_metrics = train_trainer.validate(model=model, datamodule=datamodule)
    log.info(f'val metrics: {val_metrics}')
        
    # Initialize and run the prediction
    log.info(f"Instantiating model <{config.predict._target_}>")
    predict: Predictor  = hydra.utils.instantiate(config.predict, 
                                                  model=model, 
                                                  trainer=predict_trainer, 
                                                  _recursive_=False)
    predict.process_and_save_predictions()
    log.info("Prediction process completed.")

    metric = val_metrics[0]['val/F1']
    return metric
        
@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(config: DictConfig) -> Optional[float]:
    logger.extras(config)
    logger.print_config(config)
    return train(config)

if __name__ == "__main__":
    main()
    

