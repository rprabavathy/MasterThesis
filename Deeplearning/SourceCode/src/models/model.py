import math
import torch
import numpy as np
from typing import Dict, Any
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import segmentation_models_pytorch as smp
from src import logger 

logger = logger.get_logger(__name__)


class BrainSegModel(pl.LightningModule):
    def __init__(
        self,
        architecture: str,
        encoder_name: str,
        encoder_weights: str,
        in_channels: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        activation: str = None,
        attention_type: str = None,
        auxparams: dict = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
 
        logger.info(f'Building the model : {architecture}')
        self.model = smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            activation=activation,
            attention_type=attention_type,
            **auxparams,
        )
        params = smp.encoders.get_preprocessing_params(encoder_name, pretrained=encoder_weights)
        self.num_classes = auxparams.get('classes', 3)

        self.register_buffer('std', torch.tensor(params['std']).view(1, 3, 1, 1))
        self.register_buffer('mean', torch.tensor(params['mean']).view(1, 3, 1, 1))
        logger.debug(f'Preprocessing Mean : {self.mean} and Std : {self.std}')

        self.loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE, alpha=0.5, gamma=2, reduction='mean')
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def log_precision_recall_curve(self, outputs, targets):
        """
        Calculate and log Precision-Recall Curve to TensorBoard.
         """
        batch_size, height, width = targets.shape
        targets_flat = targets.view(-1).cpu().numpy()  # Flatten to 1D
        outputs_flat = outputs.detach().cpu().numpy().reshape(batch_size, -1, outputs.shape[1])  # Flatten outputs for each class
        targets_bin = label_binarize(targets_flat, classes=[0, 1, 2])  # For multi-class, specify the classes

        for i in range(outputs.shape[1]):  # Loop over each class (background, gray matter, white matter)
            class_outputs_flat = outputs_flat[:, :, i].reshape(-1)  # Flatten class outputs for each class

            # Calculate precision-recall curve for the class
            precision, recall, _ = precision_recall_curve(targets_bin[:, i], class_outputs_flat)

            # Plot the Precision-Recall Curve for the class
            fig, ax = plt.subplots()
            ax.plot(recall, precision, label=f'Class {i}')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'Precision-Recall Curve for Class {i}')
            ax.legend()
            if self.trainer.is_global_zero:  # Ensures only one process logs
                self.trainer.logger.experiment.add_figure(f'Precision-Recall Curve Class {i}', fig, global_step=self.global_step)
            plt.close(fig)

    def forward(self, image):
        """Forward pass for the model."""
        image = (image - self.mean) / self.std
        features = self.model(image)
        return features

    def shared_step(self, batch, stage):
        image, mask = batch
        assert image.ndim == 4  # [batch_size, channels, H, W]
        mask = mask.long()
        assert mask.ndim == 3  # [batch_size, H, W]

        logits_mask = self.forward(image)  # Predict mask logits
        assert(logits_mask.shape[1] == self.num_classes)

        loss = self.loss_fn(logits_mask, mask)
        
        #if stage == 'val':
        #    self.log_precision_recall_curve(logits_mask, mask)
        
        pred_mask = logits_mask.argmax(dim=1)


        logger.debug(f'Shape-> Img: {image.shape} and {mask.shape}, Device -> Img: {image.device} and Mask: {mask.device}')
        logger.debug(f'Logits {logits_mask.shape}, device: {logits_mask.device}')

        # Compute metrics
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask, mask, mode=smp.losses.MULTICLASS_MODE, num_classes=self.num_classes
            )

        # Make sure the tensors are on the right device
        device = self.device
        tp, fp, fn, tn = tp.to(device), fp.to(device), fn.to(device), tn.to(device)

        logger.debug(f'Shared step computed for {stage}: loss={loss.item()}')

        return {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'loss': loss, 'mask': mask, 'pred_mask': pred_mask,
        }

    def shared_epoch_end(self, outputs, stage):
        """Aggregate metrics at the end of an epoch."""

        tp = torch.cat([x['tp'] for x in outputs], dim=0)
        fp = torch.cat([x['fp'] for x in outputs], dim=0)
        fn = torch.cat([x['fn'] for x in outputs], dim=0)
        tn = torch.cat([x['tn'] for x in outputs], dim=0)

        # Per-image IoU and dataset IoU calculations
        iou_dataset = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        iou_macro = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")
        acc_dataset = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro-imagewise")
        acc_macro = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        F1_dataset = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        F1_macro = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")
        recall_dataset = smp.metrics.recall(tp, fp, fn, tn, reduction="macro-imagewise")
        recall_macro = smp.metrics.recall(tp, fp, fn, tn, reduction="macro")
        precision_dataset = smp.metrics.precision(tp, fp, fn, tn, reduction="macro-imagewise")
        precision_macro = smp.metrics.precision(tp, fp, fn, tn, reduction="macro")
        
        metrics = {
            f'{stage}/iou_dataset': iou_dataset, f'{stage}/iou': iou_macro,
            f'{stage}/acc_dataset': acc_dataset, f'{stage}/acc': acc_macro,
            f'{stage}/F1_dataset': F1_dataset, f'{stage}/F1': F1_macro,
            f'{stage}/recall_dataset': recall_dataset, f'{stage}/recall': recall_macro,
            f'{stage}/precision_dataset': precision_dataset, f'{stage}/precision': precision_macro,
        }

        self.logger.experiment.add_scalars('Losses', 
                {'train_loss': self.trainer.callback_metrics.get('train/loss', 0),
                    'val_loss': self.trainer.callback_metrics.get('val/loss', 0) },
                global_step=self.current_epoch)

        self.logger.experiment.add_scalars( 'Metrics',
                {
                    'IoU': iou_macro.item(),            
                    'Accuracy': acc_macro.item(),       
                    'F1-Score': F1_macro.item(),
                    'Recall': recall_macro.item(),
                    'Precision': precision_macro.item()
                    },
                global_step=self.current_epoch)
        
        self.logger.experiment.add_scalars( 'Metrics_dataset',
                {
                    'IoU': iou_dataset.item(),
                    'Accuracy': acc_dataset.item(),
                    'F1-Score': F1_dataset.item(),
                    'Recall': recall_dataset.item(),
                    'Precision': precision_dataset.item()
                    },
                global_step=self.current_epoch)

        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        
    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, 'train')
        self.training_step_outputs.append(train_loss_info)
        self.log('train/loss', train_loss_info["loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, 'train')
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        val_loss_info = self.shared_step(batch, 'val')
        self.validation_step_outputs.append(val_loss_info)

        if self.current_epoch % 50 == 0:  # Log every 50 epochs
            images = batch[0]
            true_masks = val_loss_info['mask']
            pred_masks = val_loss_info['pred_mask']
            self.log_individual_images(images, true_masks, pred_masks, self.current_epoch, 'val')
            self.logger.experiment.flush()

        self.log('val/loss', val_loss_info["loss"], on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return val_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, 'val')
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, 'test')
        self.test_step_outputs.append(test_loss_info)

        if batch_idx % 5 == 0:
            images = batch[0]
            true_masks = test_loss_info['mask']
            pred_masks = test_loss_info['pred_mask']
            self.log_individual_images(images, true_masks, pred_masks, batch_idx, 'test')

        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, 'test')
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        self.eval()
        predImages, _ = batch
        outMask = self(predImages)
        predictedMask = outMask.argmax(dim=1)
        value_mapping = {0: 0, 1: 100, 2: 200}
        prediction = predictedMask.clone()
        for original, mapped_value in value_mapping.items():
            prediction[predictedMask == original] = mapped_value
        return prediction

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch',
                    'frequency': 1,
                },
            }
        return {'optimizer': optimizer}

    def on_fit_start(self):
        logger.debug('Model is on device: {self.device}')
    
    def log_individual_images(self, images, true_masks, pred_masks, step, stage):
        batch_size = images.size(0)
        images_per_grid = 12
        num_grids = math.ceil(batch_size / images_per_grid)

        for grid_idx in range(num_grids):
            start_idx = grid_idx * images_per_grid
            end_idx = min(start_idx + images_per_grid, batch_size)

            num_images_in_grid = end_idx - start_idx
            num_cols = 6
            num_subplots = num_images_in_grid * 3
            num_rows = math.ceil(num_subplots / num_cols)

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
            axes = np.array(axes).flatten()

            for i in range(num_images_in_grid):
                image = images[start_idx + i].cpu().permute(1, 2, 0).numpy()
                true_mask = true_masks[start_idx + i].cpu().numpy()
                pred_mask = pred_masks[start_idx + i].cpu().numpy()

                ax_idx = i * 3
                if ax_idx < len(axes):
                    axes[ax_idx].imshow(image)
                    axes[ax_idx].set_title(f'Image {start_idx + i + 1}')
                    axes[ax_idx].axis('off')

                    axes[ax_idx + 1].imshow(true_mask, cmap='gray', vmin=0, vmax=2)
                    axes[ax_idx + 1].set_title(f'True Mask {start_idx + i + 1}')
                    axes[ax_idx + 1].axis('off')

                    axes[ax_idx + 2].imshow(pred_mask, cmap='gray', vmin=0, vmax=2)
                    axes[ax_idx + 2].set_title(f'Pred Mask {start_idx + i + 1}')
                    axes[ax_idx + 2].axis('off')

            for i in range(num_images_in_grid * 3, len(axes)):
                axes[i].axis('off')

            plt.tight_layout()
            self.logger.experiment.add_figure(f'{stage}/{step}_grid_{grid_idx+1}', fig, self.current_epoch)
            plt.close(fig)

