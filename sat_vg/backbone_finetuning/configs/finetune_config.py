from dataclasses import dataclass, field
from typing import Optional, List

@dataclass
class FinetuneConfig:
    # Dataset configuration
    dataset_path: str = "/home/pokle/Trans-VG/sat_vg/backbone_finetuning/data"
    train_split: str = "train"
    val_split: str = "val"
    num_classes: int = 10  # Number of classes in RSVG dataset
    
    # Model configuration
    backbone: str = "resnet50"  # or "resnet101"
    pretrained: bool = True
    freeze_early_layers: bool = False  # Unfreeze all layers for better feature learning
    unfreeze_layers: List[str] = field(default_factory=lambda: ["layer1", "layer2", "layer3", "layer4"])
    
    # Training configuration
    batch_size: int = 16  # Smaller batch size for better generalization
    num_epochs: int = 1000  # More epochs for better convergence
    learning_rate: float = 0.00001  # Smaller learning rate
    weight_decay: float = 5e-3  # Stronger regularization
    momentum: float = 0.9
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"
    lr_step_size: int = 50  # Larger step size
    lr_gamma: float = 0.7  # More gradual decay
    warmup_epochs: int = 20  # Longer warmup
    
    # Data augmentation
    image_size: int = 224
    use_augmentation: bool = True
    
    # Hardware configuration
    num_workers: int = 4
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Logging and checkpointing
    log_dir: str = "/home/pokle/Trans-VG/sat_vg/backbone_finetuning/logs/finetune_v2"
    checkpoint_dir: str = "/home/pokle/Trans-VG/sat_vg/backbone_finetuning/checkpoints/finetune_v2"
    save_frequency: int = 5
    max_checkpoints: int = 20  # Keep more checkpoints
    checkpoint_size_limit: int = 500 * 1024 * 1024  # 500MB limit per checkpoint
    
    # Early stopping
    early_stopping_patience: int = 50  # Much longer patience
    early_stopping_min_delta: float = 0.00001  # More sensitive to small improvements 