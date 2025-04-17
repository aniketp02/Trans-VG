import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import sys
import timm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import BertModel
from .vl_transformer import build_vl_transformer
from backbone_finetuning.models.finetune_model import FineTuneModel


class SatVG(nn.Module):
    """
    Satellite Visual Grounding model
    
    Uses fine-tuned ResNet or ViT for extracting image features, BERT for text features,
    and a transformer for fusing the two modalities
    """
    def __init__(self, args):
        super(SatVG, self).__init__()
        # Configuration
        self.hidden_dim = args.hidden_dim
        self.max_query_len = args.max_query_len
        self.backbone_type = args.backbone
        
        # Load backbone for visual features
        if args.backbone == 'resnet50':
            # Try to load fine-tuned model
            finetuned_path = os.path.join('sat_vg/backbone_finetuning/checkpoints/finetune_v2', 'best_model.pt')
            if os.path.exists(finetuned_path):
                print(f"Loading fine-tuned ResNet from {finetuned_path}")
                finetuned_model = FineTuneModel.load_checkpoint(finetuned_path)
                self.backbone = finetuned_model.get_feature_extractor()
            else:
                print("Fine-tuned model not found, using pretrained ResNet50")
                self.backbone = torchvision.models.resnet50(pretrained=True)
                # Remove the final classification layer
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            
            # Set the feature dimension
            self.feature_dim = 2048
            self.num_visual_tokens = 49  # 7x7 for a standard ResNet output
            
        elif args.backbone == 'resnet101':
            # Try to load fine-tuned model
            finetuned_path = os.path.join('sat_vg/backbone_finetuning/checkpoints/finetune_v2', 'best_model.pt')
            if os.path.exists(finetuned_path):
                print(f"Loading fine-tuned ResNet from {finetuned_path}")
                finetuned_model = FineTuneModel.load_checkpoint(finetuned_path)
                self.backbone = finetuned_model.get_feature_extractor()
            else:
                print("Fine-tuned model not found, using pretrained ResNet101")
                self.backbone = torchvision.models.resnet101(pretrained=True)
                # Remove the final classification layer
                self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            
            # Set the feature dimension
            self.feature_dim = 2048
            self.num_visual_tokens = 49  # 7x7 for a standard ResNet output
            
        elif args.backbone == 'vit':
            # Load fine-tuned ViT model
            vit_checkpoints_dir = os.path.join('sat_vg/backbone_finetuning/checkpoints/vit_finetune')
            # Find the checkpoint with the highest validation accuracy
            best_checkpoint = None
            best_accuracy = 0
            
            for file in os.listdir(vit_checkpoints_dir):
                if file.startswith('checkpoint_epoch_'):
                    checkpoint_path = os.path.join(vit_checkpoints_dir, file)
                    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda'), weights_only=True)
                    accuracy = checkpoint.get('val_accuracy', 0)
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_checkpoint = checkpoint_path
            
            if best_checkpoint:
                print(f"Loading fine-tuned ViT from {best_checkpoint} (accuracy: {best_accuracy:.2f}%)")
                checkpoint = torch.load(best_checkpoint, map_location=torch.device('cuda'), weights_only=True)
                
                # Create ViT model
                self.backbone = timm.create_model(
                    'vit_base_patch16_224',
                    pretrained=False,
                    num_classes=0  # Remove classification head
                )
                
                # Load weights except the classification head
                model_dict = self.backbone.state_dict()
                pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                                  if k in model_dict and 'head' not in k}
                model_dict.update(pretrained_dict)
                self.backbone.load_state_dict(model_dict)
            else:
                print("Fine-tuned ViT model not found, using pretrained ViT")
                self.backbone = timm.create_model(
                    'vit_base_patch16_224',
                    pretrained=True,
                    num_classes=0  # Remove classification head
                )
            
            # Set the feature dimension
            self.feature_dim = 768  # ViT hidden dimension
            self.num_visual_tokens = 196  # 14x14 for ViT with 16x16 patches on 224x224 images
            
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone}")
        
        # Freeze the backbone if specified
        if args.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad_(False)
        
        # BERT for text features
        self.bert = BertModel.from_pretrained(args.bert_model)
        bert_dim = self.bert.config.hidden_size
        
        # Freeze BERT if not fine-tuning
        if args.lr_bert <= 0:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)
        
        # Linear projections to project features to the same dimension
        self.visual_projection = nn.Linear(self.feature_dim, self.hidden_dim)
        self.text_projection = nn.Linear(bert_dim, self.hidden_dim)
        
        # Learnable [REG] token
        self.reg_token = nn.Embedding(1, self.hidden_dim)
        
        # Position embeddings for the combined sequence
        # Max sequence length = 1 ([REG]) + max_query_len + visual_tokens
        total_seq_len = 1 + self.max_query_len + self.num_visual_tokens
        self.position_embedding = nn.Embedding(total_seq_len, self.hidden_dim)
        
        # Visual-linguistic transformer
        self.vl_transformer = build_vl_transformer(args)
        
        # Coordinates regression head
        self.coord_regressor = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
    
    def forward(self, image, text_tokens, text_mask):
        batch_size = image.shape[0]
        
        # Extract visual features
        if self.backbone_type in ['resnet50', 'resnet101']:
            # ResNet forward pass - shape: [B, C, H, W]
            visual_features = self.backbone(image)
            
            # Reshape visual features to sequence
            # From [B, C, H, W] to [B, H*W, C]
            visual_features = visual_features.flatten(2).permute(0, 2, 1)
        else:  # ViT
            # ViT forward pass gets patch embeddings
            visual_features = self.backbone.forward_features(image)
            
            if isinstance(visual_features, torch.Tensor):
                # If output is just the patch embeddings tensor [B, num_patches, C]
                # Remove the class token
                visual_features = visual_features[:, 1:, :]
            else:
                # If output is a dict with class_token and patch_embeddings
                visual_features = visual_features['patch_embeddings']
        
        # Project visual features to hidden dimension
        visual_features = self.visual_projection(visual_features)  # [B, tokens, hidden_dim]
        
        # Process text with BERT
        text_output = self.bert(input_ids=text_tokens, attention_mask=text_mask)
        text_features = text_output.last_hidden_state  # [B, L, bert_dim]
        
        # Project text features to hidden dimension
        text_features = self.text_projection(text_features)  # [B, L, hidden_dim]
        
        # Create [REG] token
        reg_token = self.reg_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 1, hidden_dim]
        
        # Concatenate [REG] token, text and visual features
        # [REG] + text + visual
        sequence = torch.cat([reg_token, text_features, visual_features], dim=1)  # [B, 1+L+tokens, hidden_dim]
        
        # Create position embeddings
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, device=sequence.device).unsqueeze(0).repeat(batch_size, 1)
        
        # Ensure position IDs don't exceed the embedding size
        max_position = self.position_embedding.num_embeddings - 1
        position_ids = torch.clamp(position_ids, 0, max_position)
        
        position_embeddings = self.position_embedding(position_ids)
        
        # Add position embeddings
        sequence = sequence + position_embeddings
        
        # Create attention mask (1 for tokens to attend to, 0 for padding)
        # Start with all 1s
        attn_mask = torch.ones((batch_size, seq_length), device=sequence.device)
        
        # Update with text mask (accounting for the [REG] token at the beginning)
        attn_mask[:, 1:1+text_tokens.size(1)] = text_mask.float()
        
        # Process through visual-linguistic transformer
        # We need to reshape for the transformer [B, L, C] -> [L, B, C]
        sequence = sequence.permute(1, 0, 2)
        transformed = self.vl_transformer(sequence, ~attn_mask.bool(), None)
        
        # Extract [REG] token output (first token)
        reg_output = transformed[0]  # [B, hidden_dim]
        
        # Predict bounding box coordinates (normalized [0,1])
        pred_box = self.coord_regressor(reg_output).sigmoid()  # [B, 4]
        
        return pred_box
        
    def to_cuda(self, device='cuda'):
        """Ensure all model components are on CUDA"""
        self.backbone = self.backbone.to(device)
        self.bert = self.bert.to(device)
        self.visual_projection = self.visual_projection.to(device)
        self.text_projection = self.text_projection.to(device)
        self.reg_token = self.reg_token.to(device)
        self.position_embedding = self.position_embedding.to(device)
        self.vl_transformer = self.vl_transformer.to(device)
        self.coord_regressor = self.coord_regressor.to(device)
        return self.to(device)


class MLP(nn.Module):
    """Multi-layer perceptron for coordinate regression"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x 