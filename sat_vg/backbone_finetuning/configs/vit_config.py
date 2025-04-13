from dataclasses import dataclass
from typing import Optional

@dataclass
class ViTConfig:
    # Dataset configuration
    dataset_path: str = "/home/pokle/Trans-VG/sat_vg/backbone_finetuning/data"
    train_split: str = "train"
    val_split: str = "val"
    num_classes: int = 10
    
    # Model configuration
    model_name: str = "vit_base_patch16_224"  # or "vit_large_patch16_224"
    pretrained: bool = True
    image_size: int = 224
    patch_size: int = 16
    num_layers: int = 12
    num_heads: int = 12
    hidden_dim: int = 768
    mlp_dim: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.0
    
    # Training configuration
    batch_size: int = 32
    num_epochs: int = 1000
    learning_rate: float = 0.00001
    weight_decay: float = 0.05
    warmup_epochs: int = 30
    warmup_steps: int = 5000
    
    # Learning rate scheduler
    lr_scheduler: str = "cosine"
    min_lr: float = 1e-6
    lr_gamma: float = 0.7
    
    # Data augmentation
    use_augmentation: bool = True
    mixup_alpha: float = 0.8
    cutmix_alpha: float = 1.0
    label_smoothing: float = 0.1
    
    # Hardware configuration
    num_workers: int = 4
    device: str = "cuda"
    mixed_precision: bool = True
    
    # Logging and checkpointing
    log_dir: str = "/home/pokle/Trans-VG/sat_vg/backbone_finetuning/logs/vit_finetune"
    checkpoint_dir: str = "/home/pokle/Trans-VG/sat_vg/backbone_finetuning/checkpoints/vit_finetune"
    save_frequency: int = 5
    max_checkpoints: int = 20
    
    # Early stopping
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.00001 