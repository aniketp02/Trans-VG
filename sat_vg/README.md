# SatVG: Satellite Visual Grounding

SatVG is a deep learning model for visual grounding in satellite imagery. It takes a satellite image and a natural language query as input and outputs a bounding box localizing the described region.

## Model Architecture

The model consists of three main components:

1. **Visual Encoder**
   - ResNet50/101 backbone
   - Output dimension: 2048
   - Feature map size: 7x7 (49 visual tokens)
   - Early layers frozen for efficiency

2. **Text Encoder**
   - BERT-base-uncased
   - Max query length: 40 tokens
   - Output dimension: 768
   - Optional fine-tuning

3. **Visual-Linguistic Transformer**
   - Hidden dimension: 256
   - Number of attention heads: 8
   - Feedforward dimension: 2048
   - Number of encoder layers: 6
   - Dropout: 0.2

## Input and Output

### Input
- **Image**: Tensor of shape [B, 3, H, W]
  - B: Batch size
  - 3: RGB channels
  - H, W: Height and width (default: 640x640)

- **Text**: Two tensors
  - Tokens: Shape [B, L]
  - Mask: Shape [B, L]
  - L: Max query length (40)

### Output
- **Bounding Box**: Tensor of shape [B, 4]
  - Format: [x1, y1, x2, y2]
  - Coordinates normalized to [0,1] range

## Training Process

### Loss Function
```python
class TransVGLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, focal_loss=False):
        self.bbox_loss = nn.L1Loss()  # Bounding box regression
        self.giou_loss = nn.SmoothL1Loss()  # GIoU loss
```

### Training Parameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Batch Size | 16 | Samples per batch |
| Learning Rate | 1e-4 | Initial learning rate |
| Weight Decay | 0.05 | L2 regularization |
| Epochs | 300 | Total training epochs |
| Warmup Epochs | 5 | Warmup period |

### Memory Management
```python
PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:16,garbage_collection_threshold:0.8,expandable_segments:True'
```

## Usage Example

```python
# Initialize model
model = SatVG(args)
model.to(device)

# Forward pass
pred_box = model(image, text_tokens, text_mask)

# Compute loss
loss = criterion(pred_box, target_box)
```

## Performance Metrics

1. **Mean IoU**
   - Intersection over Union between predicted and ground truth boxes

2. **Accuracy at IoU Thresholds**
   - 0.3: Loose matching
   - 0.5: Standard matching
   - 0.7: Strict matching

3. **Training Loss Components**
   - Bounding box regression loss
   - GIoU loss

## Current Model Analysis (RSVG Dataset)

### Training Performance (349 Epochs)

1. **Training Loss**
   - Initial loss: 0.267
   - Quick convergence to ~0.106 in epoch 1
   - Stabilized around 0.094-0.095
   - Final loss: 0.094
   - Early plateauing observed around epoch 50

2. **Validation Metrics**
   - Mean IoU: 0.101 (10.1%)
   - Accuracy@0.5: 0.019 (1.9%)
   - Limited improvement over training duration
   - High variance in validation performance

3. **Learning Rate Behavior**
   - Initial learning rate: 1e-4
   - BERT learning rate: peaks at 1e-5
   - Both decay to ~4.2e-7 by training end
   - Rapid decay may limit late-stage learning

4. **Training Efficiency**
   - Average epoch time: 78-82 seconds
   - Consistent training speed
   - GPU memory usage well-managed

### Areas for Improvement

1. **Learning Rate Schedule**
   - Implement slower decay schedule
   - Increase base learning rate to 2e-4
   - Extend warmup period to 10 epochs
   - Consider learning rate cycling

2. **Training Process**
   - Increase batch size for stability
   - Add gradient clipping
   - Implement early stopping
   - Validation-based LR scheduling

3. **Regularization**
   - Increase dropout rate
   - Strengthen weight decay
   - Enhance data augmentation
   - Add label smoothing

4. **Architecture Considerations**
   - Add skip connections
   - Implement feature pyramid network
   - Include auxiliary losses

## Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- transformers
- numpy
- matplotlib

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/satvg.git
cd satvg

# Install dependencies
pip install -r requirements.txt
```

## Training

```bash
# Run training script
bash run_train.sh
```

## Evaluation

```bash
# Run evaluation script
bash run_eval.sh
```

## References

1. ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
2. BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
3. Transformer: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 