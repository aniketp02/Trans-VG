#!/bin/bash

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export TORCH_USE_CUDA_DSA=1

# Add project root to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG

# Create necessary directories with absolute paths
mkdir -p /home/pokle/Trans-VG/sat_vg/backbone_finetuning/checkpoints/finetune_v2
mkdir -p /home/pokle/Trans-VG/sat_vg/backbone_finetuning/logs/finetune_v2
mkdir -p /home/pokle/Trans-VG/sat_vg/backbone_finetuning/data/train
mkdir -p /home/pokle/Trans-VG/sat_vg/backbone_finetuning/data/val

# Empty GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Run the training script as a module
cd /home/pokle/Trans-VG
python -m sat_vg.backbone_finetuning.train 