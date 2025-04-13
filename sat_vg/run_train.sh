#!/bin/bash

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_USE_CUDA_DSA=1

# Empty GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Run training script with improved configuration
python train.py \
    --backbone resnet50 \
    --hidden_dim 512 \
    --dropout 0.3 \
    --nheads 16 \
    --dim_feedforward 2048 \
    --enc_layers 6 \
    --bert_model bert-base-uncased \
    --data_root ./rsvg \
    --max_query_len 40 \
    --img_size 640 \
    --batch_size 8 \
    --lr 5e-5 \
    --lr_bert 1e-5 \
    --weight_decay 0.01 \
    --epochs 500 \
    --clip_max_norm 0.5 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --aug_blur \
    --device cuda \
    --num_workers 4 \
    --output_dir ./outputs/sat_vg \
    --warmup_epochs 20 \
    --min_lr 1e-6 \
    --label_smoothing 0.1 \
    --gradient_accumulation_steps 4 