#!/usr/bin/env python3
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import json
from tensorboard.backend.event_processing import event_accumulator

def extract_tensorboard_data(log_dir):
    """Extract data from tensorboard logs"""
    data = {}
    
    # Find all event files
    event_files = list(Path(log_dir).glob('**/events.out.tfevents.*'))
    
    for event_file in event_files:
        ea = event_accumulator.EventAccumulator(
            str(event_file),
            size_guidance={
                event_accumulator.SCALARS: 0,
                event_accumulator.IMAGES: 0,
                event_accumulator.HISTOGRAMS: 0,
            })
        ea.Reload()
        
        # Extract scalar data
        for tag in ea.Tags()['scalars']:
            events = ea.Scalars(tag)
            data[tag] = [(event.step, event.value) for event in events]
    
    return data

def plot_metrics(data, metrics, output_dir):
    """Plot specified metrics"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(12, 8))
    
    for metric in metrics:
        if metric in data:
            steps, values = zip(*data[metric])
            plt.plot(steps, values, label=metric)
    
    plt.xlabel('Steps')
    plt.ylabel('Value')
    plt.title(f"Training Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / f"{'_'.join(metrics)}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_iou_distribution(data, output_dir):
    """Plot IoU distribution"""
    if 'val/iou_distribution' not in data:
        print("No IoU distribution data found")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get the latest IoU distribution
    steps, values = zip(*data['val/iou_distribution'])
    latest_step, latest_iou = steps[-1], values[-1]
    
    plt.figure(figsize=(10, 6))
    plt.hist(latest_iou, bins=20, alpha=0.7)
    plt.axvline(x=0.5, color='r', linestyle='--', label='IoU=0.5')
    plt.xlabel('IoU')
    plt.ylabel('Count')
    plt.title(f"IoU Distribution at Step {latest_step}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "iou_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_curves(data, output_dir):
    """Plot accuracy curves for different IoU thresholds"""
    thresholds = [0.3, 0.5, 0.7]
    acc_keys = [f'val/accuracy@{t}' for t in thresholds]
    
    if not all(key in data for key in acc_keys):
        print("Not all accuracy data found")
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    plt.figure(figsize=(10, 6))
    
    for key in acc_keys:
        steps, values = zip(*data[key])
        plt.plot(steps, values, label=key)
    
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(f"Accuracy at Different IoU Thresholds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(output_dir / "accuracy_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

def main(args):
    # Extract data from tensorboard logs
    data = extract_tensorboard_data(args.log_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot training and validation loss
    plot_metrics(data, ['train/loss', 'val/loss'], output_dir)
    
    # Plot IoU
    plot_metrics(data, ['train/iou', 'val/mean_iou'], output_dir)
    
    # Plot learning rate
    plot_metrics(data, ['train/lr'], output_dir)
    
    # Plot IoU distribution
    plot_iou_distribution(data, output_dir)
    
    # Plot accuracy curves
    plot_accuracy_curves(data, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, required=True, help='Path to tensorboard logs')
    parser.add_argument('--output_dir', type=str, default='./visualizations', help='Output directory for plots')
    args = parser.parse_args()
    main(args) 