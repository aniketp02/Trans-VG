import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from transformers import BertModel
from .vl_transformer import build_vl_transformer


class SatVG(nn.Module):
    """
    Satellite Visual Grounding model
    
    Uses ResNet for extracting image features, BERT for text features,
    and a transformer for fusing the two modalities
    """
    def __init__(self, args):
        super(SatVG, self).__init__()
        # Configuration
        self.hidden_dim = args.hidden_dim
        self.max_query_len = args.max_query_len
        
        # ResNet for visual features
        if args.backbone == 'resnet50':
            self.backbone = torchvision.models.resnet50(pretrained=True)
            backbone_dim = 2048
        elif args.backbone == 'resnet101':
            self.backbone = torchvision.models.resnet101(pretrained=True)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {args.backbone}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Freeze the early layers
        for name, parameter in self.backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        
        # BERT for text features
        self.bert = BertModel.from_pretrained(args.bert_model)
        bert_dim = self.bert.config.hidden_size
        
        # Freeze BERT if not fine-tuning
        if args.lr_bert <= 0:
            for parameter in self.bert.parameters():
                parameter.requires_grad_(False)
        
        # Linear projections to project features to the same dimension
        self.visual_projection = nn.Linear(backbone_dim, self.hidden_dim)
        self.text_projection = nn.Linear(bert_dim, self.hidden_dim)
        
        # Learnable [REG] token
        self.reg_token = nn.Embedding(1, self.hidden_dim)
        
        # Position embeddings for the combined sequence
        # Max sequence length = 1 ([REG]) + max_query_len + visual_tokens
        self.num_visual_tokens = 49  # 7x7 for a standard ResNet output
        total_seq_len = 1 + self.max_query_len + self.num_visual_tokens
        self.position_embedding = nn.Embedding(total_seq_len, self.hidden_dim)
        
        # Visual-linguistic transformer
        self.vl_transformer = build_vl_transformer(args)
        
        # Coordinates regression head
        self.coord_regressor = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
    
    def forward(self, image, text_tokens, text_mask):
        batch_size = image.shape[0]
        
        # Extract visual features - shape: [B, C, H, W]
        visual_features = self.backbone(image)
        
        # Reshape visual features to sequence
        # From [B, C, H, W] to [B, H*W, C]
        visual_features = visual_features.flatten(2).permute(0, 2, 1)
        
        # Project visual features to hidden dimension
        visual_features = self.visual_projection(visual_features)  # [B, H*W, hidden_dim]
        
        # Process text with BERT
        text_output = self.bert(input_ids=text_tokens, attention_mask=text_mask)
        text_features = text_output.last_hidden_state  # [B, L, bert_dim]
        
        # Project text features to hidden dimension
        text_features = self.text_projection(text_features)  # [B, L, hidden_dim]
        
        # Create [REG] token
        reg_token = self.reg_token.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 1, hidden_dim]
        
        # Concatenate [REG] token, text and visual features
        # [REG] + text + visual
        sequence = torch.cat([reg_token, text_features, visual_features], dim=1)  # [B, 1+L+H*W, hidden_dim]
        
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