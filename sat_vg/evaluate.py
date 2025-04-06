import os
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models import build_model
from datasets import build_dataset, collate_fn
from utils.box_utils import box_cxcywh_to_xyxy, box_iou


def get_args_parser():
    """Set up argument parser for evaluation script."""
    parser = argparse.ArgumentParser('SatVG Evaluation', add_help=False)
    
    # Model parameters
    parser.add_argument('--checkpoint', required=True, type=str,
                        help="Path to model checkpoint")
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./rsvg', type=str,
                        help="Path to the dataset")
    parser.add_argument('--split', default='test', type=str,
                        choices=['train', 'val', 'test'],
                        help="Dataset split to evaluate")
    
    # Evaluation parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Batch size for evaluation")
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cuda', 'mps', 'cpu'],
                        help="Device to use for evaluation")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="Number of workers for data loading")
    parser.add_argument('--visualize', action='store_true',
                        help="Visualize predictions")
    parser.add_argument('--num_visualizations', default=10, type=int,
                        help="Number of examples to visualize")
    parser.add_argument('--output_dir', default='./outputs/sat_vg_eval',
                        help="Path to save evaluation results")
    
    return parser


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load model configuration from checkpoint
    model_args = checkpoint['args']
    
    # Update dataset-related args from current args
    model_args.data_root = args.data_root
    model_args.batch_size = args.batch_size
    model_args.num_workers = args.num_workers
    
    # Build model
    model = build_model(model_args)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Build dataset
    dataset = build_dataset(model_args, args.split)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Evaluate
    metrics = evaluate(model, data_loader, device)
    
    # Print results
    print('-' * 50)
    print("Evaluation results:")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print(f"Accuracy@0.3: {metrics['accuracy@0.3']:.4f}")
    print(f"Accuracy@0.5: {metrics['accuracy@0.5']:.4f}")
    print(f"Accuracy@0.7: {metrics['accuracy@0.7']:.4f}")
    print('-' * 50)
    
    # Save results
    results_file = output_dir / 'results.txt'
    with open(results_file, 'w') as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n")
    
    print(f"Results saved to {results_file}")
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(model, dataset, device, output_dir, args.num_visualizations)


def evaluate(model, data_loader, device):
    """
    Evaluate the model on the dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for the evaluation dataset
        device: Device to use for evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    all_ious = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Get batch data
            imgs = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(imgs, text_tokens, text_mask)
            
            # Calculate IoU
            pred_boxes = box_cxcywh_to_xyxy(outputs)
            gt_boxes = box_cxcywh_to_xyxy(targets)
            ious, _ = box_iou(pred_boxes, gt_boxes)
            diagonal_ious = torch.diag(ious)
            all_ious.append(diagonal_ious)
    
    # Calculate metrics
    all_ious = torch.cat(all_ious)
    mean_iou = all_ious.mean().item()
    
    # Calculate accuracy at different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    accu_at_threshold = {}
    for thresh in iou_thresholds:
        accu_at_threshold[f'accuracy@{thresh}'] = (all_ious > thresh).float().mean().item()
    
    # Return all metrics
    metrics = {
        'mean_iou': mean_iou,
        **accu_at_threshold
    }
    
    return metrics


def visualize_predictions(model, dataset, device, output_dir, num_samples=10):
    """
    Visualize model predictions on random samples.
    
    Args:
        model: Model to evaluate
        dataset: Dataset for visualization
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    import numpy as np
    from PIL import Image
    from torchvision import transforms
    
    # Create visualization directory
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(exist_ok=True)
    
    # Create inverse transform to convert normalized tensors back to images
    inv_normalize = transforms.Compose([
        transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        ),
        transforms.ToPILImage()
    ])
    
    # Get random indices
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    model.eval()
    with torch.no_grad():
        for idx in indices:
            # Get sample
            sample = dataset[idx]
            
            # Create batch with single sample
            batch = collate_fn([sample])
            
            # Move to device
            img = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            target = batch['target'].to(device)
            
            # Forward pass
            output = model(img, text_tokens, text_mask)
            
            # Convert to numpy arrays
            img_np = inv_normalize(img[0].cpu())
            pred_box = box_cxcywh_to_xyxy(output[0]).cpu().numpy()
            gt_box = box_cxcywh_to_xyxy(target[0]).cpu().numpy()
            
            # Scale boxes to image size
            img_size = img_np.width  # Assuming square images
            pred_box = pred_box * img_size
            gt_box = gt_box * img_size
            
            # Get text
            if isinstance(dataset.tokenizer, torch.nn.Module):
                text = dataset.tokenizer.decode(text_tokens[0])
            else:
                # If we don't have access to decode method, use a default text
                text = "Query text"
            
            # Plot image and boxes
            fig, ax = plt.subplots(1, figsize=(10, 10))
            ax.imshow(img_np)
            
            # Draw predicted box in red
            rect_pred = patches.Rectangle(
                (pred_box[0], pred_box[1]),
                pred_box[2] - pred_box[0],
                pred_box[3] - pred_box[1],
                linewidth=2,
                edgecolor='r',
                facecolor='none',
                label='Predicted'
            )
            ax.add_patch(rect_pred)
            
            # Draw ground truth box in green
            rect_gt = patches.Rectangle(
                (gt_box[0], gt_box[1]),
                gt_box[2] - gt_box[0],
                gt_box[3] - gt_box[1],
                linewidth=2,
                edgecolor='g',
                facecolor='none',
                label='Ground Truth'
            )
            ax.add_patch(rect_gt)
            
            # Add title with query text
            ax.set_title(text)
            ax.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(vis_dir / f'sample_{idx}.png')
            plt.close()
    
    print(f"Saved {num_samples} visualizations to {vis_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SatVG Evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 