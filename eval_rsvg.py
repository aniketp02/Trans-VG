#!/usr/bin/env python3
import os
import argparse
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

from configs.rsvg_config import RSVGConfig
from logger import TransVGLogger
import eval

def parse_args():
    parser = argparse.ArgumentParser('TransVG evaluation for RSVG dataset')
    
    # Model checkpoint
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='Path to the model checkpoint')
    
    # Use the RSVGConfig values as defaults
    parser.add_argument('--batch_size', type=int, default=RSVGConfig.BATCH_SIZE * 2)
    parser.add_argument('--backbone', type=str, default=RSVGConfig.BACKBONE)
    parser.add_argument('--bert_enc_num', type=int, default=RSVGConfig.BERT_ENC_NUM)
    parser.add_argument('--detr_enc_num', type=int, default=RSVGConfig.DETR_ENC_NUM)
    parser.add_argument('--dataset', type=str, default=RSVGConfig.DATASET)
    parser.add_argument('--max_query_len', type=int, default=RSVGConfig.MAX_QUERY_LEN)
    parser.add_argument('--device', type=str, default=RSVGConfig.DEVICE)
    parser.add_argument('--data_root', type=str, default=RSVGConfig.DATA_ROOT)
    parser.add_argument('--split_root', type=str, default=RSVGConfig.SPLIT_ROOT)
    
    # Evaluation set
    parser.add_argument('--eval_set', type=str, default='test', 
                        choices=['val', 'test'], help='Evaluation set')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, 
                        default=f"./results/rsvg_{RSVGConfig.BACKBONE}")
    
    # Evaluation thresholds
    parser.add_argument('--iou_thresholds', type=float, nargs='+', 
                        default=RSVGConfig.IOU_THRESHOLDS)
    
    # Visualization settings
    parser.add_argument('--viz_results', action='store_true', default=True,
                        help='Visualize results')
    parser.add_argument('--viz_count', type=int, default=100,
                        help='Number of samples to visualize')
    
    args = parser.parse_args()
    return args

def visualize_results(results, output_dir, count=100):
    """Visualize evaluation results"""
    output_dir = Path(output_dir)
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics summary
    metrics = results['metrics']
    plt.figure(figsize=(10, 6))
    plt.bar(metrics.keys(), metrics.values())
    plt.ylabel('Value')
    plt.title('Evaluation Metrics')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_summary.png", dpi=300)
    plt.close()
    
    # IoU distribution
    ious = results['all_ious']
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=20, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='IoU=0.5')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title(f"IoU Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / "iou_distribution.png", dpi=300)
    plt.close()
    
    # Visualize individual results
    if 'examples' in results and len(results['examples']) > 0:
        for i, example in enumerate(results['examples'][:count]):
            img = example['image']
            gt_box = example['gt_box']
            pred_box = example['pred_box']
            query = example['query']
            iou = example['iou']
            
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            
            # Draw ground truth box in green
            x1, y1, x2, y2 = gt_box
            width, height = x2 - x1, y2 - y1
            plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, 
                                            fill=False, edgecolor='green', 
                                            linewidth=2, label='Ground Truth'))
            
            # Draw predicted box in blue/red
            x1, y1, x2, y2 = pred_box
            width, height = x2 - x1, y2 - y1
            color = 'blue' if iou > 0.5 else 'red'
            plt.gca().add_patch(plt.Rectangle((x1, y1), width, height, 
                                            fill=False, edgecolor=color, 
                                            linewidth=2, label='Prediction'))
            
            plt.title(f"Query: {query}\nIoU: {iou:.3f}")
            plt.legend()
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(viz_dir / f"example_{i:04d}.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set up output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize logger
    logger = TransVGLogger(output_dir, vars(args))
    
    # Call the evaluation script
    results = eval.main(args, logger)
    
    # Save results to disk
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Visualize results
    if args.viz_results:
        visualize_results(results, output_dir, args.viz_count)
    
    logger.close()
    print(f"Evaluation completed. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 