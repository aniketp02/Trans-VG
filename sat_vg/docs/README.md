# SatVG Documentation

This directory contains documentation for the SatVG (Satellite Visual Grounding) model, experiments, and future improvement strategies.

## Contents

- [ViT Backbone Experiments](vit_backbone_experiments.md) - Documentation of experiments using Vision Transformer as a backbone
- [Improvement Strategies](improvement_strategies.md) - Potential strategies for enhancing model performance

## Project Overview

SatVG is a model for visual grounding in satellite imagery, which aims to localize objects or regions in satellite images based on natural language descriptions. The model combines visual features from a backbone network (ResNet or ViT) with textual features from BERT, and uses a transformer architecture to fuse these modalities for accurate localization.

## Model Architecture

The SatVG architecture consists of three main components:

1. **Visual Backbone** - Extracts features from satellite images (ResNet50/101 or ViT)
2. **Text Encoder** - Processes natural language descriptions using BERT
3. **Multimodal Fusion** - Combines visual and textual features through a transformer

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Mean IoU** - Average Intersection over Union between predicted and ground truth boxes
- **Accuracy@0.3** - Percentage of predictions with IoU > 0.3
- **Accuracy@0.5** - Percentage of predictions with IoU > 0.5
- **Accuracy@0.7** - Percentage of predictions with IoU > 0.7 