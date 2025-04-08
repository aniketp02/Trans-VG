import os
import math
import time
import torch
import random
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn.utils import clip_grad_norm_

from models import build_model
from datasets import build_dataset, collate_fn
from utils.logger import Logger
from utils.box_utils import box_iou, box_cxcywh_to_xyxy
from utils.loss_utils import TransVGLoss


def get_args_parser():
    """Set up argument parser for command line arguments."""
    parser = argparse.ArgumentParser('SatVG Training', add_help=False)
    
    # Model parameters
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.2, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help="Name of the BERT model to use")
    
    # Dataset parameters
    parser.add_argument('--data_root', default='./rsvg', type=str,
                        help="Path to the dataset")
    parser.add_argument('--max_query_len', default=40, type=int,
                        help="Maximum length of the query")
    parser.add_argument('--img_size', default=640, type=int,
                        help="Size of the input image")
    parser.add_argument('--use_augmentation', action='store_true',
                        help="Whether to use data augmentation")
    
    # Training parameters
    parser.add_argument('--batch_size', default=16, type=int,
                        help="Batch size for training")
    parser.add_argument('--gradient_accumulation_steps', default=4, type=int,
                        help="Number of gradient accumulation steps")
    parser.add_argument('--lr', default=1e-4, type=float,
                        help="Learning rate")
    parser.add_argument('--lr_bert', default=1e-5, type=float,
                        help="Learning rate for BERT")
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help="Weight decay")
    parser.add_argument('--epochs', default=300, type=int,
                        help="Number of epochs")
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help="Number of warmup epochs")
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help="Minimum learning rate")
    parser.add_argument('--clip_max_norm', default=1.0, type=float,
                        help="Maximum norm for gradient clipping")
    parser.add_argument('--label_smoothing', default=0.1, type=float,
                        help="Label smoothing factor")
    parser.add_argument('--eval_interval', default=1, type=int,
                        help="Interval for evaluation during training")
    parser.add_argument('--save_interval', default=5, type=int,
                        help="Interval for saving checkpoints")
    
    # Augmentation parameters
    parser.add_argument('--aug_crop', action='store_true',
                        help="Use random crop augmentation")
    parser.add_argument('--aug_scale', action='store_true',
                        help="Use multi-scale augmentation")
    parser.add_argument('--aug_translate', action='store_true',
                        help="Use random translate augmentation")
    parser.add_argument('--aug_blur', action='store_true',
                        help="Use gaussian blur augmentation")
    parser.add_argument('--aug_color', action='store_true',
                        help="Use color jittering augmentation")
    parser.add_argument('--aug_erase', action='store_true',
                        help="Use random erasing augmentation")
    
    # Runtime parameters
    parser.add_argument('--device', default='cuda', type=str,
                        help="Device to use for training (cuda, mps, or cpu)")
    parser.add_argument('--seed', default=42, type=int,
                        help="Random seed")
    parser.add_argument('--num_workers', default=4, type=int,
                        help="Number of workers for data loading")
    parser.add_argument('--output_dir', default='./outputs/sat_vg', type=str,
                        help="Directory to save outputs")
    parser.add_argument('--resume', default='', type=str,
                        help="Path to checkpoint to resume from")
    
    return parser


def create_lr_scheduler(optimizer, num_epochs, warmup_epochs, min_lr):
    """Create learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return epoch / warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


def main(args):
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = torch.device('cpu')
        else:
            # Set CUDA device to GPU 0
            torch.cuda.set_device(1)
            device = torch.device('cuda:1')
            
            # Configure memory settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set memory allocation settings
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:16,garbage_collection_threshold:0.8,expandable_segments:True'
            
            print(f"Using CUDA device {torch.cuda.current_device()}")
            print(f"CUDA device name: {torch.cuda.get_device_name()}")
            print(f"CUDA device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
            
            # Enable mixed precision training
            scaler = torch.cuda.amp.GradScaler()
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        scaler = None
    else:
        device = torch.device('cpu')
        scaler = None
    
    print(f"Using device: {device}")
    
    # Verify CUDA is working
    if device.type == 'cuda':
        print("\nCUDA Status:")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        print(f"CUDA device properties: {torch.cuda.get_device_properties(torch.cuda.current_device())}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Empty cache before starting
        torch.cuda.empty_cache()
        print(f"After clearing cache - CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"After clearing cache - CUDA memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
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
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    data_loader_val = DataLoader(
        dataset_val, 
        batch_size=args.batch_size,
        sampler=sampler_val,
        drop_last=False, 
        collate_fn=collate_fn, 
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Build model
    print("Building model...")
    model = build_model(args)
    model.to(device)
    
    # Enable gradient checkpointing for transformer layers
    if hasattr(model, 'vl_transformer') and hasattr(model.vl_transformer, 'encoder'):
        for layer in model.vl_transformer.encoder.layers:
            layer.use_checkpoint = True
    
    # Create parameter groups with different learning rates
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() 
                   if "bert" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() 
                   if "bert" in n and p.requires_grad],
         "lr": args.lr_bert},
    ]
    
    # Create optimizer with weight decay
    optimizer = optim.AdamW(param_dicts, lr=args.lr,
                           weight_decay=args.weight_decay)
    
    # Create learning rate scheduler
    lr_scheduler = create_lr_scheduler(optimizer, args.epochs,
                                     args.warmup_epochs, args.min_lr)
    
    # Optionally resume from checkpoint
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['scheduler'])
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
            logger=logger,
            scaler=scaler
        )
        
        # Update learning rate
        lr_scheduler.step()
        
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
                    scheduler=lr_scheduler,
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
                scheduler=lr_scheduler,
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
        scheduler=lr_scheduler,
        epoch=args.epochs - 1,
        args=args,
        output_dir=output_dir,
        filename='final_model.pth'
    )
    
    print("Training completed!")


def train_one_epoch(model, data_loader, optimizer, device, epoch, args, logger, scaler=None):
    """Train the model for one epoch."""
    model.train()
    criterion = TransVGLoss(
        label_smoothing=args.label_smoothing,
        focal_loss=True  # Enable focal loss for better handling of class imbalance
    )
    criterion.to(device)
    
    running_loss = 0.0
    num_batches = len(data_loader)
    start_time = time.time()
    
    # Print CUDA memory info at start of epoch
    if device.type == 'cuda':
        print(f"\nEpoch {epoch} - Initial CUDA memory:")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Empty cache at the start of each epoch
        torch.cuda.empty_cache()
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(data_loader):
        # Get batch data
        imgs = batch['img'].to(device, non_blocking=True)
        text_tokens = batch['text_tokens'].to(device, non_blocking=True)
        text_mask = batch['text_mask'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        # Print CUDA memory info for first batch
        if i == 0 and device.type == 'cuda':
            print(f"\nFirst batch - CUDA memory after data transfer:")
            print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
        
        # Forward pass with mixed precision
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(imgs, text_tokens, text_mask)
                loss_dict = criterion(outputs, targets)
                loss = loss_dict['loss']
                # Scale loss by gradient accumulation steps
                loss = loss / args.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.clip_max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                
                # Update weights
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Clear memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        else:
            # Forward pass
            outputs = model(imgs, text_tokens, text_mask)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict['loss']
            # Scale loss by gradient accumulation steps
            loss = loss / args.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights if we've accumulated enough gradients
            if (i + 1) % args.gradient_accumulation_steps == 0:
                # Gradient clipping
                if args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                
                # Update weights
                optimizer.step()
                optimizer.zero_grad()
                
                # Clear memory
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Update running loss (use the unscaled loss)
        running_loss += loss.item() * args.gradient_accumulation_steps
        
        # Print progress and CUDA memory info periodically
        if (i + 1) % 10 == 0:
            if device.type == 'cuda':
                print(f"Epoch {epoch}, Batch {i+1}/{num_batches}, Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")
                print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            else:
                print(f"Epoch {epoch}, Batch {i+1}/{num_batches}, Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")
    
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
    parser = get_args_parser()
    args = parser.parse_args()
    main(args) 