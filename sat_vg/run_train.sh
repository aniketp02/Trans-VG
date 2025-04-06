#!/bin/bash

# Run SatVG training with default parameters
python train.py \
    --backbone resnet50 \
    --hidden_dim 256 \
    --dropout 0.1 \
    --nheads 8 \
    --dim_feedforward 2048 \
    --enc_layers 6 \
    --bert_model bert-base-uncased \
    --data_root ./rsvg \
    --max_query_len 40 \
    --img_size 640 \
    --use_augmentation \
    --batch_size 16 \
    --lr 1e-4 \
    --lr_bert 1e-5 \
    --weight_decay 1e-4 \
    --epochs 100 \
    --lr_drop 60 \
    --clip_max_norm 0.1 \
    --eval_interval 1 \
    --save_interval 10 \
    --device mps \
    --num_workers 4 \
    --output_dir ./outputs/sat_vg 