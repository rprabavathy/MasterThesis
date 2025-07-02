# 🧠 Deep Learning-Based Brain Tissue Segmentation in Blockface Images

This repository contains the complete work for my **Master's thesis**, completed as part of the *Computer Simulation in Science* program at **Bergische Universität Wuppertal**, Germany.

The repository includes implementations for both classical and deep learning-based segmentation techniques, as well as performance evaluation metrics for assessing the segmentation quality.

## 📁 Directory Structure

- **`Conventional/`**  
  Contains segmentation methods based on **classical image processing algorithms**, such as thresholding, clustering, and morphological operations.

- **`DeepLearning/`**  
  Includes segmentation approaches based on **deep learning**, using two different **sampling strategies** applied to train neural networks on blockface brain images.
  This part of the work follows a modular layout using a `src/` directory, and is configured using the **[Hydra](https://hydra.cc)** framework for flexible experiment management.
  Key features:
  - Configuration files are stored in the `config/` folder
  - Training, validation, and inference logic is inside `src/`
  - Easily switch models, datasets, and hyperparameters via YAML configs

- **`Metrics/`**  
  Provides scripts for **evaluating segmentation performance**, including:
  - **Hausdorff Distance (HD)**
  - **F1 Score**
  - **Jaccard scores**
  - **Precision**
  - **Recall**
  - **Boundary overlap plots for visualizations**
  -
  Includes radar plot for visualizing the performance metrics of implemented Algorithm

- **`DL_Segmented/`**
  Includes post processing of masks predicted using **deep learning** and segmentation of blockface brain images.
