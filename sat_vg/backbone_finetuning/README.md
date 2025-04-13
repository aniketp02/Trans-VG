# ResNet Backbone Fine-tuning

This module provides tools for fine-tuning ResNet models on satellite imagery datasets. The fine-tuned models can then be used as feature extractors in the main SatVG model.

## Directory Structure

```
backbone_finetuning/
├── configs/           # Configuration files
├── data/             # Dataset and data loading code
├── models/           # Model definitions
├── utils/            # Utility functions
└── train.py          # Training script
```

## Setup

1. Install dependencies:
```bash
pip install torch torchvision tensorboard tqdm
```

2. Prepare your dataset:
   - Organize your satellite images in the following structure:
   ```
   dataset_root/
   ├── train/
   │   ├── class1/
   │   │   ├── image1.jpg
   │   │   └── ...
   │   └── class2/
   │       ├── image1.jpg
   │       └── ...
   └── val/
       ├── class1/
       │   ├── image1.jpg
       │   └── ...
       └── class2/
           ├── image1.jpg
           └── ...
   ```

## Configuration

Modify `configs/finetune_config.py` to set your training parameters:

```python
config = FinetuneConfig(
    dataset_path="path/to/your/dataset",
    backbone="resnet50",  # or "resnet101"
    num_classes=1000,     # Adjust based on your dataset
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    # ... other parameters
)
```

Key parameters:
- `freeze_early_layers`: Whether to freeze early layers of ResNet
- `unfreeze_layers`: List of layer names to unfreeze for fine-tuning
- `lr_scheduler`: Learning rate scheduler type ("cosine", "step", or "plateau")
- `early_stopping_patience`: Number of epochs to wait before early stopping

## Training

Run the training script:
```bash
python -m sat_vg.backbone_finetuning.train
```

The script will:
1. Load and preprocess the dataset
2. Initialize the ResNet model with pretrained weights
3. Fine-tune the model on your satellite images
4. Save checkpoints and log training metrics

## Monitoring

Training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir logs/finetune
```

## Using the Fine-tuned Model

After training, you can load the fine-tuned model and use it as a feature extractor:

```python
from sat_vg.backbone_finetuning.models.finetune_model import FineTuneModel

# Load the fine-tuned model
model = FineTuneModel.load_checkpoint("checkpoints/finetune/best_model.pt")

# Get the feature extractor
feature_extractor = model.get_feature_extractor()
```

## Best Practices

1. Start with a small learning rate (e.g., 0.001) when fine-tuning
2. Use data augmentation to improve generalization
3. Monitor validation loss to prevent overfitting
4. Save checkpoints regularly to resume training if needed
5. Use early stopping to prevent overfitting

## Troubleshooting

1. If training is too slow:
   - Reduce batch size
   - Use mixed precision training
   - Increase number of workers

2. If model is not learning:
   - Check learning rate
   - Verify data preprocessing
   - Ensure proper layer freezing/unfreezing

3. If overfitting:
   - Increase data augmentation
   - Add regularization
   - Reduce model capacity

# Backbone Model Experiments

## ResNet Experiments

### Experiment 1: Initial Configuration
- **Model**: ResNet50
- **Configuration**:
  - Batch size: 64
  - Learning rate: 0.0005
  - Weight decay: 1e-3
  - Epochs: 300
  - Frozen layers: layer1
  - Unfrozen layers: layer3, layer4
- **Results**:
  - Training Loss: ~0.41
  - Validation Loss: ~1.11
  - Validation Accuracy: 65%
  - Early stopping triggered
- **Analysis**: Model showed signs of overfitting with significant gap between training and validation loss.

### Experiment 2: Improved Configuration
- **Model**: ResNet50
- **Configuration**:
  - Batch size: 16
  - Learning rate: 0.00001
  - Weight decay: 5e-3
  - Epochs: 500
  - Unfrozen layers: layer2, layer3, layer4
- **Results**:
  - Training Loss: ~0.67-0.71
  - Validation Loss: ~1.05-1.08
  - Validation Accuracy: 64.25%
  - Early stopping triggered
- **Analysis**: Despite stronger regularization and more unfrozen layers, model still struggled with overfitting and limited accuracy.

## Rationale for Moving to Vision Transformer

### Limitations of ResNet for Satellite Imagery
1. **Architectural Constraints**:
   - ResNet's convolutional nature limits its ability to capture long-range dependencies
   - Struggles with multi-scale features common in satellite imagery
   - Local receptive fields may miss global context

2. **Performance Issues**:
   - Plateaued at ~65% accuracy despite multiple configurations
   - Persistent overfitting (training loss << validation loss)
   - Limited improvement with increased model capacity

3. **Feature Extraction Challenges**:
   - Satellite images require understanding of:
     - Global spatial relationships
     - Multi-scale features
     - Complex texture patterns
     - Long-range dependencies

### Advantages of Vision Transformer
1. **Architectural Benefits**:
   - Self-attention mechanism for global feature extraction
   - Better handling of long-range dependencies
   - More effective at multi-scale feature learning
   - Flexible receptive fields through attention

2. **Expected Improvements**:
   - Better accuracy for satellite imagery tasks
   - More robust feature extraction
   - Better handling of complex spatial relationships
   - Reduced overfitting through attention mechanisms

## Next Steps
1. Create new branch for ViT experiments
2. Implement and train ViT backbone
3. Compare performance with ResNet baseline
4. Fine-tune ViT configuration based on results

## Metrics Comparison
```
Model         | Training Loss | Validation Loss | Accuracy | Overfitting Gap
-------------|---------------|-----------------|----------|----------------
ResNet (Exp1) | 0.41         | 1.11           | 65%      | 0.70
ResNet (Exp2) | 0.67-0.71    | 1.05-1.08      | 64.25%   | 0.37-0.41
```

The significant gap between training and validation metrics, combined with the plateau in accuracy, suggests that ResNet's architecture may not be optimal for satellite imagery feature extraction. The Vision Transformer's ability to capture global context and handle multi-scale features makes it a promising alternative. 