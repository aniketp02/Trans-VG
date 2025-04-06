import os
import time
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_norm_

from models import build_model
from datasets import build_dataset, collate_fn
from utils.logger import Logger
from utils.box_utils import box_iou, box_cxcywh_to_xyxy


def get_args_parser():
    """Set up argument parser for command line arguments."""
    parser = argparse.ArgumentParser('SatVG Training', add_help=False)
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str, 
                        choices=['resnet50', 'resnet101'],
                        help="Backbone for the visual model")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Hidden dimension of the model")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout probability")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Dimension of the feedforward network")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoder layers")
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help="BERT model name")
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./rsvg', type=str,
                        help="Path to the dataset")
    parser.add_argument('--max_query_len', default=40, type=int,
                        help="Maximum query length")
    parser.add_argument('--img_size', default=640, type=int,
                        help="Image size")
    parser.add_argument('--use_augmentation', action='store_true',
                        help="Whether to use data augmentation")
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Batch size for training")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="Learning rate")
    parser.add_argument('--lr_bert', default=1e-5, type=float,
                        help="Learning rate for BERT")
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help="Weight decay")
    parser.add_argument('--epochs', default=100, type=int,
                        help="Number of epochs")
    parser.add_argument('--lr_drop', default=60, type=int,
                        help="Epoch at which to drop learning rate")
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help="Maximum norm for gradient clipping")
    parser.add_argument('--eval_interval', default=1, type=int,
                        help="Interval for evaluation during training")
    parser.add_argument('--save_interval', default=10, type=int,
                        help="Interval for saving checkpoints")
    
    # Runtime parameters
    parser.add_argument('--device', default='cuda', type=str,
                        choices=['cuda', 'mps', 'cpu'],
                        help="Device to use for training")
    parser.add_argument('--seed', default=42, type=int,
                        help="Random seed")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="Number of workers for data loading")
    parser.add_argument('--output_dir', default='./outputs/sat_vg',
                        help="Path to save outputs")
    parser.add_argument('--resume', default='', type=str,
                        help="Path to resume from checkpoint")
    
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
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize logger
    logger = Logger(output_dir)
    logger.log_params(vars(args))
    
    # Build datasets
    print("Building datasets...")
    dataset_train = build_dataset(args, 'train')
    dataset_val = build_dataset(args, 'val')
    
    # Create samplers and data loaders
    sampler_train = RandomSampler(dataset_train)
    sampler_val = SequentialSampler(dataset_val)
    
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)
    
    data_loader_train = DataLoader(
        dataset_train, 
        batch_sampler=batch_sampler_train,
        collate_fn=collate_fn, 
        num_workers=args.num_workers
    )
    
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers
    )
    
    # Build model
    print("Building model...")
    model = build_model(args)
    model.to(device)
    
    # Create parameter groups with different learning rates
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                   if "bert" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() 
                   if "bert" in n and p.requires_grad],
         "lr": args.lr_bert},
    ]
    
    # Create optimizer and scheduler
    optimizer = optim.AdamW(param_dicts, lr=args.lr,
                           weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.lr_drop, gamma=0.1)
    
    # Optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at {args.resume}")
    
    # Print model information
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Training loop
    print("Starting training...")
    best_val_iou = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch}/{args.epochs - 1}")
        
        # Train for one epoch
        train_stats = train_one_epoch(
            model=model,
            data_loader=data_loader_train,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            logger=logger
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        if (epoch + 1) % args.eval_interval == 0:
            val_stats = evaluate(
                model=model,
                data_loader=data_loader_val,
                device=device,
                args=args,
                logger=logger,
                epoch=epoch
            )
            
            # Log validation stats
            for k, v in val_stats.items():
                logger.log_scalar(f"val/{k}", v, epoch)
            
            # Save best model
            if val_stats['mean_iou'] > best_val_iou:
                best_val_iou = val_stats['mean_iou']
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    args=args,
                    output_dir=output_dir,
                    filename='best_model.pth'
                )
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                output_dir=output_dir,
                filename=f'checkpoint_{epoch:04d}.pth'
            )
        
        # Log training stats
        for k, v in train_stats.items():
            logger.log_scalar(f"train/{k}", v, epoch)
    
    # Save final model
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=args.epochs - 1,
        args=args,
        output_dir=output_dir,
        filename='final_model.pth'
    )
    
    print("Training completed!")


def train_one_epoch(model, data_loader, optimizer, device, epoch, args, logger):
    """Train the model for one epoch."""
    model.train()
    criterion = torch.nn.MSELoss()
    
    running_loss = 0.0
    num_batches = len(data_loader)
    start_time = time.time()
    
    for i, batch in enumerate(data_loader):
        # Get batch data
        imgs = batch['img'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_mask = batch['text_mask'].to(device)
        targets = batch['target'].to(device)
        
        # Forward pass
        outputs = model(imgs, text_tokens, text_mask)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if args.clip_max_norm > 0:
            clip_grad_norm_(model.parameters(), args.clip_max_norm)
        
        # Update weights
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Epoch {epoch}, Batch {i+1}/{num_batches}, Loss: {loss.item():.4f}")
    
    # Calculate metrics
    avg_loss = running_loss / num_batches
    end_time = time.time()
    epoch_time = end_time - start_time
    
    print(f"Epoch {epoch} completed in {epoch_time:.2f}s, Avg Loss: {avg_loss:.4f}")
    
    return {
        'loss': avg_loss,
        'lr': optimizer.param_groups[0]['lr'],
        'lr_bert': optimizer.param_groups[1]['lr'],
        'epoch_time': epoch_time
    }


def evaluate(model, data_loader, device, args, logger, epoch):
    """Evaluate the model on the validation set."""
    model.eval()
    criterion = torch.nn.MSELoss()
    
    total_loss = 0.0
    all_ious = []
    num_batches = len(data_loader)
    
    with torch.no_grad():
        for batch in data_loader:
            # Get batch data
            imgs = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            targets = batch['target'].to(device)
            
            # Forward pass
            outputs = model(imgs, text_tokens, text_mask)
            loss = criterion(outputs, targets)
            
            # Update total loss
            total_loss += loss.item()
            
            # Calculate IoU
            pred_boxes = box_cxcywh_to_xyxy(outputs)
            gt_boxes = box_cxcywh_to_xyxy(targets)
            ious, _ = box_iou(pred_boxes, gt_boxes)
            diagonal_ious = torch.diag(ious)
            all_ious.append(diagonal_ious)
    
    # Calculate metrics
    avg_loss = total_loss / num_batches
    all_ious = torch.cat(all_ious)
    mean_iou = all_ious.mean().item()
    
    # Calculate accuracy at different IoU thresholds
    iou_thresholds = [0.3, 0.5, 0.7]
    accu_at_threshold = {}
    for thresh in iou_thresholds:
        accu_at_threshold[f'accuracy@{thresh}'] = (all_ious > thresh).float().mean().item()
    
    # Print results
    print('-' * 50)
    print(f"Validation results for epoch {epoch}:")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    for thresh, acc in accu_at_threshold.items():
        print(f"{thresh}: {acc:.4f}")
    print('-' * 50)
    
    # Return all metrics
    metrics = {
        'loss': avg_loss,
        'mean_iou': mean_iou,
        **accu_at_threshold
    }
    
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, args, output_dir, filename):
    """Save a checkpoint of the model."""
    checkpoint_path = output_dir / filename
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
        'args': args
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser('SatVG Training', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args) 