#!/bin/bash

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_USE_CUDA_DSA=1
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG

# Create output directories
OUTPUT_DIR="outputs/satvg_vit_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR/checkpoints
mkdir -p $OUTPUT_DIR/logs

echo "Using ViT as the backbone feature extractor for SatVG"
echo "Output directory: $OUTPUT_DIR"

# Run training script with correct path
python train.py \
    --backbone vit \
    --batch_size 16 \
    --lr 5e-5 \
    --lr_bert 1e-5 \
    --epochs 100 \
    --output_dir $OUTPUT_DIR \
    --device cuda \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --warmup_epochs 5 \
    --min_lr 1e-6 \
    --clip_max_norm 1.0 