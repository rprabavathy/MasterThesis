import cv2
import albumentations as A
from src import logger 
logger = logger.get_logger(__name__) 

def trainTransform(enable=True, flip=True, affine=None, clahe=0.3, gauss_noise=None, elasticTransform=None, rCrop=None):
    """
    This function returns the augmentation pipeline based on the YAML configuration.
    """
    logger.info("Applying the Transform")
    train_transform = []

    if enable:
        if flip:
            logger.info("Adding HorizontalFlip and VerticalFlip with p=0.5")
            train_transform.append(A.HorizontalFlip(p=0.5))
            train_transform.append(A.VerticalFlip(p=0.5))

        if affine:
            logger.info(f"Adding Affine transform with parameters: {affine}")
            train_transform.append(A.Affine(
                scale=affine.get("scale", [0.9, 1.1]),
                rotate=affine.get("rotate", [-30, 30]),
                translate_percent=affine.get("translate_percent", [-0.05, 0.05]),
                shear=affine.get("shear", [-0.2, 0.2]),
                interpolation=getattr(cv2, affine.get("interpolation", "INTER_LINEAR")),
                p=1.0,
            ))

        if clahe:
            logger.info(f"Adding CLAHE with p={clahe}")
            train_transform.append(A.CLAHE(p=clahe))
        if gauss_noise:
            logger.info(f"Adding Gaussian Noise with parameters: {gauss_noise}")
            train_transform.append(A.GaussNoise(
                std_range=gauss_noise.get("std_range", (0.2, 0.44)),
                mean_range=gauss_noise.get("mean_range", (0.0, 0.0)),
                per_channel=gauss_noise.get("per_channel", True),
                noise_scale_factor=gauss_noise.get("noise_scale_factor", 0.5),
                p=gauss_noise.get("p", 0.5)
            ))
        if elasticTransform: 
            logger.info(f"Applying Elastic Transform")
            train_transform.append(A.ElasticTransform(
                alpha=elasticTransform.get("alpha", 1.0),
                sigma=elasticTransform.get("sigma", 10),
                p=elasticTransform.get("p",0.5)
            ))
        
        if rCrop:
            logger.info("random crop")
            train_transform.append(A.RandomCrop(
                height=rCrop.get("height", 64), 
                width=rCrop.get("weight", 64), 
                p=rCrop.get("p",0.5)
            ))         
    return A.Compose(train_transform)
