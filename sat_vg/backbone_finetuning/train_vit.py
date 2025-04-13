import os
import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import timm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler

from .configs.vit_config import ViTConfig
from .data.satellite_dataset import SatelliteDataset


def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def create_dataloaders(config: ViTConfig) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_dataset = SatelliteDataset(
        root_dir=config.dataset_path,
        split=config.train_split,
        image_size=config.image_size
    )
    
    val_dataset = SatelliteDataset(
        root_dir=config.dataset_path,
        split=config.val_split,
        image_size=config.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_model(config: ViTConfig) -> nn.Module:
    """Create Vision Transformer model."""
    model = timm.create_model(
        config.model_name,
        pretrained=config.pretrained,
        num_classes=config.num_classes,
        drop_rate=config.dropout,
        drop_path_rate=0.1,
        img_size=config.image_size,
        patch_size=config.patch_size,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        embed_dim=config.hidden_dim,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_attention_rate=config.attention_dropout
    )
    return model


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    mixup_fn: Optional[Mixup],
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
        
        optimizer.zero_grad()
        
        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(val_loader, desc="Validation"):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return total_loss / len(val_loader), accuracy


def main(config: ViTConfig) -> None:
    """Main training function."""
    # Setup
    logger = setup_logging(config.log_dir)
    device = torch.device(config.device)
    
    # Create model
    model = create_model(config)
    model = model.to(device)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(config)
    
    # Create optimizer
    optimizer = create_optimizer_v2(
        model,
        opt='adamw',
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Create learning rate scheduler
    num_steps = len(train_loader) * config.num_epochs
    lr_scheduler, _ = create_scheduler(
        optimizer,
        num_epochs=config.num_epochs,
        num_steps=num_steps,
        warmup_epochs=config.warmup_epochs,
        warmup_steps=config.warmup_steps,
        min_lr=config.min_lr
    )
    
    # Create loss function
    criterion = LabelSmoothingCrossEntropy(smoothing=config.label_smoothing)
    
    # Create mixup augmentation
    mixup_fn = None
    if config.use_augmentation:
        mixup_fn = Mixup(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            label_smoothing=config.label_smoothing,
            num_classes=config.num_classes
        )
    
    # Create gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            device, mixup_fn, scaler
        )
        
        # Validate
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        lr_scheduler.step(epoch + 1)
        
        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}: "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.2f}%"
        )
        
        # Check for early stopping
        if val_loss < best_val_loss - config.early_stopping_min_delta:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            os.makedirs(config.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_accuracy
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                logger.info("Early stopping triggered")
                break
    
    logger.info("Training completed")


if __name__ == "__main__":
    config = ViTConfig()
    main(config) 