#!/bin/bash

# Run SatVG evaluation with default parameters
python evaluate.py \
    --checkpoint ./outputs/sat_vg/best_model.pth \
    --data_root ./rsvg \
    --split test \
    --batch_size 16 \
    --device mps \
    --num_workers 4 \
    --visualize \
    --num_visualizations 10 \
    --output_dir ./outputs/sat_vg_eval 