# SVHN Dataset Curation Tool

This repository contains tools for creating a balanced subset of the SVHN (Street View House Numbers) dataset through manual annotation and curation.

## Overview

The SVHN dataset contains over 600,000 digit images, but for many research purposes, a smaller, carefully curated subset is more useful. This project provides:

1. **Manual Annotation Tool**: An interactive OpenCV-based tool for manually selecting high-quality images
2. **Balanced Dataset Creation**: Scripts to ensure equal representation across all digit classes (0-9)
3. **Train/Validation/Test Splits**: Organized splits for machine learning workflows

## Features

- Interactive visual annotation interface
- Real-time class balance monitoring
- Configurable target samples per class
- Automated train/validation/test splitting
- Balanced dataset generation

## Annotation Tool Interface

![SVHN Annotator Tool](SVHN_annotator.png)

The annotation tool displays:
- Current image being reviewed
- Image label
- Real-time count of selected samples per digit class
- Progress indicators (✓ marks when target reached)

### Controls
- **ENTER**: Select current image
- **ESC**: Skip current image
- **Q**: Quit annotation session

## Usage

### 1. Manual Annotation

Run the annotation tool to manually select images:

```bash
python main.py
```

This will:
- Load the SVHN training dataset
- Display images one by one for manual review
- Allow you to select/skip images based on quality
- Track progress toward target samples per class
- Save selected indices to `selected_svhn_indices.txt`

### 2. Create Balanced Dataset

Use the Jupyter notebook `visualize.ipynb` to:
- Load your manually selected indices
- Create a balanced subset (equal samples per class)
- Visualize class distribution
- Generate train/validation/test splits

### 3. Dataset Splits

The process generates three balanced splits:
- **Training**: 100 samples per class (1,000 total)
- **Validation**: 20 samples per class (200 total)
- **Test**: 30 samples per class (300 total)

## Generated Files

### Index Files
- [`train_balanced_indices.txt`](train_balanced_indices.txt) - Training set indices
- [`val_balanced_indices.txt`](val_balanced_indices.txt) - Validation set indices  
- [`test_balanced_indices.txt`](test_balanced_indices.txt) - Test set indices
- [`balanced_svhn_indices_150.txt`](balanced_svhn_indices_150.txt) - All balanced indices (150 per class)

### Dataset Files
- `balanced_svhn_150_per_class.pth` - PyTorch tensor containing the balanced dataset

## Requirements

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

## Dataset Structure

The curated dataset provides:
- **1,500 total images** (150 per digit class)
- **High-quality manual selection** through visual inspection
- **Balanced representation** across all digit classes
- **Clean train/val/test splits** for reproducible experiments

## File Structure

```
├── main.py                          # Manual annotation tool
├── visualize.ipynb                  # Dataset analysis and splitting
├── data/
│   └── train_32x32.mat             # Original SVHN data
├── train_balanced_indices.txt       # Training indices
├── val_balanced_indices.txt         # Validation indices
├── test_balanced_indices.txt        # Test indices
├── balanced_svhn_indices_150.txt    # All balanced indices
├── balanced_svhn_150_per_class.pth  # PyTorch dataset
└── SVHN_annotator.png              # Tool screenshot
```

## Configuration

You can modify the annotation targets in `main.py`:

```python
target_per_class = 300  # Adjust samples per class
resize_dims = (300, 300)  # Display image size
```

## Results

The manual curation process ensures:
- **Quality Control**: Only visually clear, properly cropped digits
- **Class Balance**: Exactly equal representation per digit
- **Manageable Size**: Focused dataset for efficient training/testing
- **Reproducibility**: Fixed indices for consistent experiments

## Citation

If you use this curated SVHN subset in your research, please cite the original SVHN paper:

```
@inproceedings{37648,title = {Reading Digits in Natural Images with Unsupervised Feature Learning},author = {Yuval Netzer and Tao Wang and Adam Coates and Alessandro Bissacco and Bo Wu and Andrew Y. Ng},year = {2011},URL = {http://ufldl.stanford.edu/housenumbers/nips2011_housenumbers.pdf},booktitle = {NIPS Workshop on Deep Learning and Unsupervised Feature Learning 2011}}

```
