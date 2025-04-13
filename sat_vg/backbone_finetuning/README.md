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