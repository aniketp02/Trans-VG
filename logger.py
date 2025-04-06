import os
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import datetime
import json
from PIL import Image, ImageDraw

class TransVGLogger:
    def __init__(self, log_dir, config=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # Save configuration
        if config is not None:
            with open(self.log_dir / 'config.json', 'w') as f:
                json.dump(config, f, indent=2)
        
        # Create directories for visualizations
        self.viz_dir = self.log_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        print(f"Logger initialized at {self.log_dir}")
    
    def log_metrics(self, metrics, step, prefix=''):
        """Log metrics to tensorboard"""
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item() if v.numel() == 1 else v.mean().item()
            self.writer.add_scalar(f'{prefix}{k}', v, step)
    
    def log_images(self, images, step, prefix=''):
        """Log images to tensorboard"""
        for k, v in images.items():
            self.writer.add_image(f'{prefix}{k}', v, step, dataformats='HWC')
    
    def log_histogram(self, values, step, name):
        """Log histogram to tensorboard"""
        self.writer.add_histogram(name, values, step)
    
    def log_pr_curve(self, labels, predictions, step, name):
        """Log precision-recall curve to tensorboard"""
        self.writer.add_pr_curve(name, labels, predictions, step)
    
    def log_grounding_result(self, image, gt_box, pred_box, query, is_correct, step, prefix=''):
        """Visualize grounding result and save to disk"""
        # Create a copy of the image
        img = image.copy()
        draw = ImageDraw.Draw(img)
        
        # Draw ground truth box in green
        draw.rectangle(gt_box.tolist(), outline='green', width=2)
        
        # Draw predicted box in blue if correct, red if incorrect
        color = 'blue' if is_correct else 'red'
        draw.rectangle(pred_box.tolist(), outline=color, width=2)
        
        # Add query text
        draw.text((10, 10), query, fill='white')
        
        # Save visualization
        img_path = self.viz_dir / f'{prefix}_{step}.jpg'
        img.save(img_path)
        
        # Also log to tensorboard
        self.writer.add_image(f'{prefix}/result_{step}', np.array(img), step, dataformats='HWC')
    
    def close(self):
        self.writer.close() 