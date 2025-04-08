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
    --hidden_dim 256 \
    --dropout 0.2 \
    --nheads 8 \
    --dim_feedforward 2048 \
    --enc_layers 6 \
    --bert_model bert-base-uncased \
    --data_root ./rsvg \
    --max_query_len 40 \
    --img_size 640 \
    --use_augmentation \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr 1e-4 \
    --lr_bert 1e-5 \
    --weight_decay 0.05 \
    --epochs 400 \
    --warmup_epochs 5 \
    --min_lr 1e-6 \
    --clip_max_norm 1.0 \
    --label_smoothing 0.1 \
    --aug_crop \
    --aug_scale \
    --aug_translate \
    --aug_blur \
    --aug_color \
    --aug_erase \
    --device cuda \
    --num_workers 4 \
    --output_dir ./outputs/sat_vg \
    --save_interval 5 \
    --eval_interval 1 