import torch
import torch.nn as nn
import torchvision.models as models
from typing import List, Optional
import os
import logging


class FineTuneModel(nn.Module):
    """
    Model for fine-tuning ResNet on satellite images.
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = True,
        freeze_early_layers: bool = True,
        unfreeze_layers: Optional[List[str]] = None
    ):
        """
        Args:
            backbone: ResNet backbone to use ("resnet50" or "resnet101")
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
            freeze_early_layers: Whether to freeze early layers
            unfreeze_layers: List of layer names to unfreeze for fine-tuning
        """
        super().__init__()
        
        # Initialize tracking attributes
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        
        # Load pretrained ResNet
        if backbone == "resnet50":
            self.model = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet101":
            self.model = models.resnet101(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify the final fully connected layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        
        # Freeze layers if specified
        if freeze_early_layers:
            self._freeze_layers(unfreeze_layers)
    
    def _freeze_layers(self, unfreeze_layers: Optional[List[str]] = None):
        """
        Freeze all layers except those specified in unfreeze_layers.
        
        Args:
            unfreeze_layers: List of layer names to keep unfrozen
        """
        if unfreeze_layers is None:
            unfreeze_layers = []
        
        for name, param in self.model.named_parameters():
            # Check if this parameter belongs to any of the unfrozen layers
            should_unfreeze = any(layer in name for layer in unfreeze_layers)
            param.requires_grad = should_unfreeze
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Output tensor of shape [B, num_classes]
        """
        return self.model(x)
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get the feature extractor part of the model (everything except the final FC layer).
        
        Returns:
            Feature extractor module
        """
        return nn.Sequential(
            *list(self.model.children())[:-1],
            nn.Flatten()
        )
    
    def save_checkpoint(self, path: str, logger: logging.Logger) -> None:
        """Save model checkpoint with error handling."""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save to temporary file first
            temp_path = path + '.tmp'
            
            # Save checkpoint to temporary file
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'best_val_loss': self.best_val_loss,
                'best_val_acc': self.best_val_acc,
            }, temp_path)
            
            # If successful, rename temp file to final file
            if os.path.exists(path):
                os.remove(path)
            os.rename(temp_path, path)
            
            logger.info(f"Checkpoint saved successfully to {path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {str(e)}")
            # Clean up temporary file if it exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise
    
    @classmethod
    def load_checkpoint(cls, path: str) -> 'FineTuneModel':
        """
        Load model from checkpoint.
        
        Args:
            path: Path to the checkpoint
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path)
        model = cls(
            backbone=checkpoint['backbone'].lower(),
            num_classes=checkpoint['num_classes']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 