"""
Configuration for RSVG dataset training with TransVG
"""

import os
from datetime import datetime

class RSVGConfig:
    # Basic settings
    DATASET = 'rsvg'
    DATA_ROOT = './rsvg'
    SPLIT_ROOT = './rsvg'
    
    # Model settings
    BACKBONE = 'resnet50'  # Options: resnet50, resnet101, vit
    BERT_ENC_NUM = 12
    DETR_ENC_NUM = 6
    MAX_QUERY_LEN = 40
    
    # Training settings
    BATCH_SIZE = 8  # Adjust based on GPU memory
    LEARNING_RATE = 1e-4
    LR_BERT = 1e-5
    WEIGHT_DECAY = 1e-4
    EPOCHS = 100
    NUM_WORKERS = 2  # Adjust based on CPU cores
    
    # Optimizer and scheduler
    OPTIMIZER = 'adamw'  # Options: adamw, adam, sgd
    LR_SCHEDULER = 'step'  # Options: step, poly, halfdecay, cosine
    LR_DROP = 20
    
    # Augmentation
    USE_AUG_CROP = True
    USE_AUG_SCALE = True
    USE_AUG_TRANSLATE = True
    USE_AUG_BLUR = False
    
    # Device
    DEVICE = 'mps'  # Options: cuda, mps, cpu
    
    # Logging and checkpointing
    RUN_NAME = f"transvg_rsvg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    OUTPUT_DIR = f"./outputs/{RUN_NAME}"
    LOG_INTERVAL = 100  # Log every N batches
    SAVE_INTERVAL = 5  # Save checkpoint every N epochs
    VIZ_INTERVAL = 200  # Visualize predictions every N batches
    
    # Metrics
    IOU_THRESHOLDS = [0.3, 0.5, 0.7]  # IoU thresholds for evaluation
    
    # Satellite-specific parameters
    SCALE_FACTOR = 1.0  # Scale factor for satellite images
    
    @staticmethod
    def get_training_command():
        cmd = (
            f"python train.py "
            f"--batch_size {RSVGConfig.BATCH_SIZE} "
            f"--lr {RSVGConfig.LEARNING_RATE} "
            f"--lr_bert {RSVGConfig.LR_BERT} "
            f"--epochs {RSVGConfig.EPOCHS} "
            f"--backbone {RSVGConfig.BACKBONE} "
            f"--bert_enc_num {RSVGConfig.BERT_ENC_NUM} "
            f"--detr_enc_num {RSVGConfig.DETR_ENC_NUM} "
            f"--dataset {RSVGConfig.DATASET} "
            f"--max_query_len {RSVGConfig.MAX_QUERY_LEN} "
            f"--output_dir {RSVGConfig.OUTPUT_DIR} "
            f"--device {RSVGConfig.DEVICE} "
            f"--data_root {RSVGConfig.DATA_ROOT} "
            f"--split_root {RSVGConfig.SPLIT_ROOT} "
            f"--optimizer {RSVGConfig.OPTIMIZER} "
            f"--lr_scheduler {RSVGConfig.LR_SCHEDULER} "
            f"--lr_drop {RSVGConfig.LR_DROP} "
            f"--weight_decay {RSVGConfig.WEIGHT_DECAY} "
            f"--num_workers {RSVGConfig.NUM_WORKERS} "
        )
        
        if RSVGConfig.USE_AUG_CROP:
            cmd += " --aug_crop"
        if RSVGConfig.USE_AUG_SCALE:
            cmd += " --aug_scale"
        if RSVGConfig.USE_AUG_TRANSLATE:
            cmd += " --aug_translate"
        if RSVGConfig.USE_AUG_BLUR:
            cmd += " --aug_blur"
            
        return cmd 