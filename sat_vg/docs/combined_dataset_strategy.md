# Combined Dataset Strategy for Improved ViT Backbone

## Overview

This document outlines the approach for using multiple satellite imagery datasets to train the Vision Transformer (ViT) backbone for the SatVG model. By combining RSVG with additional satellite datasets (DIOR-RSVG and OPT-RSVG), we aim to significantly improve the feature extraction capabilities of the backbone, leading to better visual grounding performance.

## Implementation Date: April 17, 2025

## Motivation

The original ViT backbone trained solely on the RSVG dataset achieved a validation accuracy of around 68%, which is good but leaves substantial room for improvement. Vision Transformers are known to benefit significantly from larger and more diverse datasets.

By expanding the training data, we expect to:
1. Improve feature extraction capabilities
2. Increase robustness to different imaging conditions
3. Better handle the diversity of objects and contexts in satellite imagery
4. Push the accuracy ceiling to 75-80%+ for the backbone

## Datasets Used

The combined dataset leverages three satellite imagery sources:

1. **RSVG**: The original Satellite Visual Grounding dataset
   - Location: `/home/pokle/Trans-VG/sat_vg/backbone_finetuning/data`
   - Content: Classified satellite imagery organized by object categories

2. **DIOR-RSVG**: DIOR (Dataset for Object Detection in Optical Remote Sensing Images) adapted for visual grounding
   - Location: `/home/pokle/Trans-VG/sat_vg/dior-rsvg`
   - Content: High-resolution satellite imagery with various objects

3. **OPT-RSVG**: Additional optical satellite imagery dataset
   - Location: `/home/pokle/Trans-VG/sat_vg/opt-rsvg`
   - Content: Optical satellite images covering diverse terrain and objects

## Implementation Details

### 1. Dataset Preparation

We created a `DatasetProcessor` class that:
- Combines images from all three datasets
- Preserves category structure
- Automatically infers categories from directory structure or filenames
- Calculates appropriate normalization statistics
- Splits data into training and validation sets (80/20 split)
- Saves the combined dataset to a new location

### 2. Training Configuration

Modified ViT training configuration for the combined dataset:
- Increased batch size (64 vs. 32)
- Higher learning rate (0.0001 vs. 0.00001)
- Reduced number of epochs (100 vs. 1000)
- Shortened warmup period (10 vs. 30 epochs)
- Enabled mixup and cutmix augmentation
- Set early stopping patience to 20 epochs

### 3. Model Integration

Updated the SatVG model to:
- First check for the combined dataset ViT model
- Use it if available, falling back to the original model if not
- Report the validation accuracy from the checkpoint

## Running the Pipeline

The entire pipeline can be executed with a single command:

```bash
./sat_vg/backbone_finetuning/run_vit_combined.sh
```

This script:
1. Prepares the combined dataset
2. Trains the ViT model
3. Provides instructions for using the trained model with SatVG

## Results

Expected outcomes (to be updated after training):
- Combined dataset size: [TBD] images
- Number of categories: [TBD]
- Training time: Approximately [TBD] hours
- Best validation accuracy: Expected 75-80%+ (versus 68% for original)

## Limitations and Future Work

- Current implementation doesn't leverage semantic annotations from datasets
- Category mapping between datasets could be improved
- Potential for self-supervised pre-training on unlabeled satellite imagery
- Future work could explore larger models (ViT-Large) with this expanded dataset 