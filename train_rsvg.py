#!/usr/bin/env python3
import os
import argparse
import torch
import random
import numpy as np
from pathlib import Path

from configs.rsvg_config import RSVGConfig
from logger import TransVGLogger
import train

def parse_args():
    parser = argparse.ArgumentParser('TransVG training for RSVG dataset')
    
    # Use the RSVGConfig values as defaults
    parser.add_argument('--batch_size', type=int, default=RSVGConfig.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=RSVGConfig.LEARNING_RATE)
    parser.add_argument('--lr_bert', type=float, default=RSVGConfig.LR_BERT)
    parser.add_argument('--epochs', type=int, default=RSVGConfig.EPOCHS)
    parser.add_argument('--backbone', type=str, default=RSVGConfig.BACKBONE)
    parser.add_argument('--bert_enc_num', type=int, default=RSVGConfig.BERT_ENC_NUM)
    parser.add_argument('--detr_enc_num', type=int, default=RSVGConfig.DETR_ENC_NUM)
    parser.add_argument('--dataset', type=str, default=RSVGConfig.DATASET)
    parser.add_argument('--max_query_len', type=int, default=RSVGConfig.MAX_QUERY_LEN)
    parser.add_argument('--output_dir', type=str, default=RSVGConfig.OUTPUT_DIR)
    parser.add_argument('--device', type=str, default=RSVGConfig.DEVICE)
    parser.add_argument('--data_root', type=str, default=RSVGConfig.DATA_ROOT)
    parser.add_argument('--split_root', type=str, default=RSVGConfig.SPLIT_ROOT)
    
    # Logging parameters
    parser.add_argument('--log_interval', type=int, default=RSVGConfig.LOG_INTERVAL)
    parser.add_argument('--save_interval', type=int, default=RSVGConfig.SAVE_INTERVAL)
    parser.add_argument('--viz_interval', type=int, default=RSVGConfig.VIZ_INTERVAL)
    
    # Augmentation flags
    parser.add_argument('--aug_crop', action='store_true', default=RSVGConfig.USE_AUG_CROP)
    parser.add_argument('--aug_scale', action='store_true', default=RSVGConfig.USE_AUG_SCALE)
    parser.add_argument('--aug_translate', action='store_true', default=RSVGConfig.USE_AUG_TRANSLATE)
    parser.add_argument('--aug_blur', action='store_true',
                        help="If true, use gaussian blur augmentation")
    
    # Evaluation thresholds
    parser.add_argument('--iou_thresholds', type=float, nargs='+', default=RSVGConfig.IOU_THRESHOLDS)
    
    # For frozen weights
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    
    # Add seed parameter
    parser.add_argument('--seed', default=13, type=int, help='Random seed')
    
    # Vision-Language Transformer parameters
    parser.add_argument('--vl_dropout', default=0.1, type=float,
                        help="Dropout applied in the vision-language transformer")
    parser.add_argument('--vl_nheads', default=8, type=int,
                        help="Number of attention heads inside the vision-language transformer's attentions")
    parser.add_argument('--vl_hidden_dim', default=256, type=int,
                        help='Size of the embeddings (dimension of the vision-language transformer)')
    parser.add_argument('--vl_dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the vision-language transformer blocks")
    parser.add_argument('--vl_enc_layers', default=6, type=int,
                        help='Number of encoders in the vision-language transformer')
    
    # DETR parameters
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                       help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=0, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--imsize', default=640, type=int, help='image size')
    parser.add_argument('--emb_size', default=512, type=int,
                        help='fusion module embedding dimensions')
    
    # Others
    parser.add_argument('--lr_visu_cnn', default=1e-5, type=float)
    parser.add_argument('--lr_visu_tra', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_power', default=0.9, type=float, help='lr poly power')
    parser.add_argument('--clip_max_norm', default=0., type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval', dest='eval', default=False, action='store_true', help='if evaluation only')
    parser.add_argument('--optimizer', default='adamw', type=str)
    parser.add_argument('--lr_scheduler', default='step', type=str)
    parser.add_argument('--lr_drop', default=60, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--detr_model', default=None, type=str, help='detr model')
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str, help='bert model')
    parser.add_argument('--light', dest='light', default=False, action='store_true', help='if use smaller model')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--model_name', type=str, default='TransVG',
                        help="Name of model to be exploited.")
    
    args = parser.parse_args()
    return args

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = TransVGLogger(output_dir, vars(args))
    
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Call the original training script with our arguments
    train.main(args, logger)
    
    # Create visualizations
    os.system(f"python visualize_results.py --log_dir {output_dir}/tensorboard --output_dir {output_dir}/plots")
    
    logger.close()
    print(f"Training completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 