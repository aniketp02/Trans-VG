# SatVG: Satellite Visual Grounding

SatVG is a model for visual grounding on satellite imagery. The model takes an image and a text query as input and outputs a bounding box localizing the described region in the image.

## Architecture

The model architecture consists of:

1. **Visual Encoder**: ResNet-50/101 for extracting image features
2. **Text Encoder**: BERT for encoding text queries
3. **Visual-Linguistic Transformer**: For fusing visual and textual features
4. **Learnable [REG] Token**: A special token that attends to both modalities
5. **Coordinate Regression**: MLP that predicts the bounding box coordinates

![SatVG Architecture](architecture.png)

## Requirements

- Python 3.8+
- PyTorch 1.10+
- transformers
- matplotlib
- numpy

You can install the required packages using pip:

```bash
pip install torch torchvision transformers matplotlib numpy
```

## Dataset

The model is designed to work with the RSVG (Remote Sensing Visual Grounding) dataset, which consists of satellite images paired with text queries and bounding box annotations.

The dataset should be structured as follows:

```
rsvg/
├── images/             # Directory containing satellite images
├── rsvg_train.pth      # Training data
├── rsvg_val.pth        # Validation data
└── rsvg_test.pth       # Test data
```

## Training

To train the model:

```bash
# Make the script executable
chmod +x run_train.sh

# Run training
./run_train.sh
```

You can customize the training parameters by modifying the script or passing them directly:

```bash
python train.py \
    --backbone resnet50 \
    --hidden_dim 256 \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_bert 1e-5 \
    --epochs 100 \
    --device mps \
    --output_dir ./outputs/sat_vg
```

## Evaluation

To evaluate the model:

```bash
# Make the script executable
chmod +x run_eval.sh

# Run evaluation
./run_eval.sh
```

You can customize the evaluation parameters:

```bash
python evaluate.py \
    --checkpoint ./outputs/sat_vg/best_model.pth \
    --data_root ./rsvg \
    --split test \
    --visualize \
    --output_dir ./outputs/sat_vg_eval
```

## Results

The model is evaluated using the following metrics:

- **Mean IoU**: Average Intersection over Union
- **Accuracy@0.3**: Percentage of predictions with IoU > 0.3
- **Accuracy@0.5**: Percentage of predictions with IoU > 0.5
- **Accuracy@0.7**: Percentage of predictions with IoU > 0.7

## Citation

If you use this model in your research, please cite:

```
@article{satvg2023,
  title={SatVG: Visual Grounding on Satellite Imagery},
  author={Your Name},
  journal={arXiv preprint},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details. 