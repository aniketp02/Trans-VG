import os
import logging
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import glob

from sat_vg.backbone_finetuning.configs.finetune_config import FinetuneConfig
from sat_vg.backbone_finetuning.data.satellite_dataset import SatelliteDataset
from sat_vg.backbone_finetuning.models.finetune_model import FineTuneModel


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


def create_dataloaders(config: FinetuneConfig) -> Tuple[DataLoader, DataLoader]:
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


def create_optimizer(model: nn.Module, config: FinetuneConfig) -> optim.Optimizer:
    """Create optimizer for training."""
    # Get parameters that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    return optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )


def create_scheduler(
    optimizer: optim.Optimizer,
    config: FinetuneConfig,
    num_steps: int
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """Create learning rate scheduler."""
    if config.lr_scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_steps
        )
    elif config.lr_scheduler == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.lr_gamma,
            patience=config.lr_step_size
        )
    return None


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)
        
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


def save_checkpoint(model: FineTuneModel, epoch: int, config: FinetuneConfig, logger: logging.Logger) -> None:
    """Save model checkpoint with error handling and disk space check."""
    try:
        # Check available disk space
        checkpoint_dir = config.checkpoint_dir
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            
        # Get available disk space
        stat = os.statvfs(checkpoint_dir)
        available_space = stat.f_bavail * stat.f_frsize
        
        # Estimate checkpoint size (roughly)
        estimated_size = 500 * 1024 * 1024  # 500MB estimate
        
        if available_space < estimated_size:
            logger.warning(f"Low disk space: {available_space / (1024*1024):.2f}MB available, "
                          f"estimated checkpoint size: {estimated_size / (1024*1024):.2f}MB")
            return
            
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        model.save_checkpoint(checkpoint_path, logger)
        
        # Keep only the last N checkpoints
        checkpoints = sorted(glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt')))
        if len(checkpoints) > config.max_checkpoints:
            for old_checkpoint in checkpoints[:-config.max_checkpoints]:
                try:
                    os.remove(old_checkpoint)
                except Exception as e:
                    logger.error(f"Error removing old checkpoint {old_checkpoint}: {str(e)}")
                    
    except Exception as e:
        logger.error(f"Error in save_checkpoint: {str(e)}")
        # Don't raise the exception, just log it and continue training


def main(config: FinetuneConfig) -> None:
    """Main training function with improved error handling."""
    try:
        # Setup logging
        logger = setup_logging(config.log_dir)
        logger.info("Starting training with config: %s", config)
        
        # Check disk space before starting
        stat = os.statvfs(config.checkpoint_dir)
        available_space = stat.f_bavail * stat.f_frsize
        if available_space < 1024 * 1024 * 1024:  # Less than 1GB
            logger.warning(f"Low disk space: {available_space / (1024*1024*1024):.2f}GB available")
            
        # Initialize model and data
        model = FineTuneModel(
            backbone=config.backbone,
            num_classes=config.num_classes,
            pretrained=config.pretrained,
            freeze_early_layers=config.freeze_early_layers,
            unfreeze_layers=config.unfreeze_layers
        ).to(torch.device(config.device))
        train_loader, val_loader = create_dataloaders(config)
        
        # Create optimizer and criterion
        optimizer = create_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        
        # Create learning rate scheduler
        scheduler = create_scheduler(
            optimizer,
            config,
            num_steps=len(train_loader) * config.num_epochs
        )
        
        # Set up mixed precision training if enabled
        scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Set up TensorBoard
        writer = SummaryWriter(config.log_dir)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config.num_epochs):
            try:
                # Update current epoch
                model.current_epoch = epoch
                
                # Train
                train_loss = train_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    torch.device(config.device),
                    scaler
                )
                
                # Validate
                val_loss, val_accuracy = validate(
                    model,
                    val_loader,
                    criterion,
                    torch.device(config.device)
                )
                
                # Update learning rate
                if scheduler is not None:
                    if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()
                
                # Log metrics
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/val', val_loss, epoch)
                writer.add_scalar('Accuracy/val', val_accuracy, epoch)
                
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train Loss: {train_loss:.4f}, "
                    f"Val Loss: {val_loss:.4f}, "
                    f"Val Accuracy: {val_accuracy:.2f}%"
                )
                
                # Save checkpoint
                if epoch % config.save_frequency == 0:
                    save_checkpoint(model, epoch, config, logger)
                
                # Early stopping check
                if val_loss < best_val_loss - config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    model.best_val_loss = val_loss
                    model.best_val_acc = val_accuracy
                    patience_counter = 0
                    # Save best model
                    best_model_path = os.path.join(config.checkpoint_dir, 'best_model.pt')
                    model.save_checkpoint(best_model_path, logger)
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        logger.info("Early stopping triggered")
                        break
                    
            except Exception as e:
                logger.error(f"Error in epoch {epoch}: {str(e)}")
                # Continue to next epoch
                continue
                
        writer.close()
        logger.info("Training completed")
        
    except Exception as e:
        logger.error(f"Fatal error in training: {str(e)}")
        raise


if __name__ == "__main__":
    config = FinetuneConfig()
    main(config) 