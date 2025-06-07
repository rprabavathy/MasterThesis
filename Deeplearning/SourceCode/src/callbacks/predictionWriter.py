import os
import numpy as np
from PIL import Image
from pytorch_lightning import callbacks

from src import logger
log = logger.get_logger(__name__)

class CustomWriter(callbacks.BasePredictionWriter):
    def __init__(self, output_dir, write_interval="epoch"):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        """This method is called at the end of each epoch. Save predictions for each rank (GPU)."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        for batch in batch_indices:
            for group_idx, group in enumerate(batch):
                log.debug(f"Processing group {group_idx}, group : {group}")   
                predbatch = predictions[group_idx]
                log.debug(f'length of image_pred : {len(predbatch)}')
                for idx, image_pred in zip(group,predbatch):
                    # Get the padding info from the dataset (this was saved earlier)
                    padding_info = trainer.datamodule.dataset[idx][1]
                    pad_h, pad_w = padding_info["pad_h"], padding_info["pad_w"]

                    pred_cpu = image_pred.cpu().numpy()
                    pred_cpu = pred_cpu[:pred_cpu.shape[0] - pad_h, :pred_cpu.shape[1] - pad_w]  # Unpad the prediction
                    log.debug(f'Shape after Prediction : {pred_cpu.shape}')
                    image_file = os.path.basename(trainer.datamodule.dataset.image_paths[idx])
                    last_folder = os.path.basename(os.path.normpath(trainer.datamodule.dataset.image_dir))
                    log.info(f'last_folder : {last_folder}')
                    relative_path = os.path.relpath(os.path.dirname(trainer.datamodule.dataset.image_paths[idx]),
                                       trainer.datamodule.dataset.image_dir)
                    output_subdir = os.path.join(self.output_dir, last_folder, relative_path)

                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir, exist_ok=True)

                    result_mask_path = os.path.join(output_subdir, f"{os.path.splitext(image_file)[0]}.png")
                    predicted_mask_image = Image.fromarray(pred_cpu.astype(np.uint8))
                    predicted_mask_image.save(result_mask_path)

                    log.info(f"Rank {trainer.global_rank}: Saved predicted mask for {image_file} at {result_mask_path}") 

