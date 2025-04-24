# Satellite Visual Grounding Experiments

## Current Experiment: Satellite ViT with Improved Training (2025-04-22)

### Implemented Improvements

1. **Satellite-specialized Vision Transformer backbone**
   - Integrated SatMAE pretrained weights specifically trained on satellite imagery
   - End-to-end fine-tuning (not freezing backbone) to allow adaptation to our specific task

2. **Enhanced Data Augmentation Pipeline**
   - Random horizontal and vertical flips
   - Random resized crop (scale 0.7-1.0, ratio 0.75-1.33)
   - Random affine transformations (translation ±20%)
   - Color jitter (brightness: 0.6-1.4, contrast: 0.6-1.4, saturation: 0.6-1.4, hue: ±0.1)
   - Gaussian blur (kernel size: 5×5, sigma: 0.1-2.0)

3. **Improved Hyperparameters**
   - Batch size of 8 with gradient accumulation (8 steps) for effective batch size of 64
   - Learning rate of 5e-5 for the model and 1e-5 for BERT encoder
   - Cosine learning rate schedule with 10 epochs warmup period
   - Minimum learning rate of 1e-7
   - Gradient clipping at 1.0
   - Weight decay of 0.05

4. **Better Monitoring and Validation**
   - Detailed validation metrics after every 5 epochs
   - Memory usage tracking
   - Model checkpointing (best model + epoch checkpoints)

### Results Analysis (Epoch 9)
- Validation Loss: 0.0949
- Mean IoU: 0.1090
- Accuracy@0.3: 0.1055
- Accuracy@0.5: 0.0268
- Accuracy@0.7: 0.0016

Despite our improvements, the model shows limited accuracy in precisely localizing objects in satellite imagery based on textual descriptions.

## Dataset Improvements (2025-04-23)

### Identified Dataset Issues

1. **Limited Data Volume**
   - Current RSVG dataset only contains 2,967 training samples
   - Insufficient data diversity for the model to generalize well
   - Small dataset makes model prone to overfitting

2. **Data Distribution**
   - Limited object categories (primarily focused on sports fields, bridges, tanks)
   - Imbalanced distribution of reference types
   - Spatial references often ambiguous or imprecise

3. **Query Complexity**
   - Simple queries mostly using positional references
   - Limited attribute-based descriptions (color, size, shape)
   - Few relational descriptions (relative to other objects)

### Combined Dataset Approach

To address the dataset limitations, we've implemented a combined dataset approach:

1. **Multiple Dataset Integration**
   - Original RSVG dataset: 2,967 train, 635 validation, 637 test samples
   - DIOR-RSVG dataset: Remote sensing dataset with detailed object annotations
   - OPT-RSVG dataset: Additional satellite imagery with referring expressions

2. **Unified Data Format**
   - Converted all datasets to a common format matching RSVG structure
   - Each sample consists of: [image_name, None, [x1, y1, x2, y2], text_query, None]
   - All images copied to a common directory structure

3. **Data Conversion Pipeline**
   - Created dataset conversion tool in `datasets/convert_datasets.py`
   - Automatic conversion from XML annotation format
   - Combined dataset available in `combined-rsvg` directory

4. **Expected Benefits**
   - 2-3x increase in training data volume
   - Greater diversity of objects, relationships, and query types
   - Improved generalization across different satellite imagery sources
   - More balanced training examples for different reference types

## Planned Future Improvements

1. **Architectural Enhancements**
   - Smaller patch size (8×8 instead of 16×16) for finer-grained feature extraction
   - Multi-scale feature integration to capture objects at different scales
   - Instance-specific attention mechanisms to focus on relevant regions based on text query
   - Feature pyramid networks for better spatial understanding

2. **Training Strategy Refinements**
   - Curriculum learning (start with easier examples, gradually introduce harder ones)
   - Progressive resizing (start training at lower resolution, gradually increase)
   - Mixed precision training for faster iterations
   - Longer training schedule with custom learning rate decay

3. **Data Enhancements**
   - Additional data augmentation techniques specific to satellite imagery
   - Dataset expansion or synthetic data generation
   - Class-balanced sampling to address potential dataset imbalances

4. **Regularization Techniques**
   - Feature dropout for better generalization
   - Layer-wise learning rate decay
   - Contrastive learning objectives

5. **Evaluation Improvements**
   - Analyze performance by query type and object size
   - Visual debugging of model predictions to identify common failure patterns

## ViT Backbone Experiment (2023-04-23)

### Configuration
- Backbone: sat_vit
- Training on combined dataset
- Epochs: 200
- Learning rate: 5e-05
- Batch size: 8
- Image size: 224

### Results
After 134 epochs of training:
- Loss: 2.081 (minimal decrease from 2.087)
- Mean IoU: ~0.049 (5%)
- Accuracy@0.3: ~5.8%
- Accuracy@0.5: ~1.5% 
- Accuracy@0.7: ~0.15%

### Observations
The Vision Transformer backbone showed very limited learning on the satellite visual grounding task:
- Metrics remained almost flat throughout training
- No significant improvement in performance over 134 epochs
- The model struggled to effectively localize objects based on textual descriptions

### Next Steps
1. Try using the pre-finetuned ResNet backbone (`/home/pokle/Trans-VG/sat_vg/backbone_finetuning/checkpoints/finetune_v2/best_model.pt`) with the combined dataset
2. Consider exploring specialized architectures for satellite imagery that better capture spatial relationships
3. Investigate multi-scale feature extraction approaches to better handle variable object sizes in overhead imagery 