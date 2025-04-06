import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class VLTransformer(nn.Module):
    """
    Visual-Linguistic Transformer for satellite visual grounding.
    
    This transformer processes the combined sequence of [REG] token,
    text tokens, and visual tokens.
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Create transformer encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        # Create encoder with multiple layers
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Initialize parameters
        self._reset_parameters()
        
        # Store model dimensions
        self.d_model = d_model
        self.nhead = nhead
    
    def _reset_parameters(self):
        """Initialize the weights using Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, mask, pos=None):
        """
        Forward pass through the transformer.
        
        Args:
            src: Input tensor of shape [seq_len, batch_size, d_model]
            mask: Boolean mask for padding (True for padding) of shape [batch_size, seq_len]
            pos: Optional position embeddings of shape [seq_len, batch_size, d_model]
        
        Returns:
            Output tensor of shape [seq_len, batch_size, d_model]
        """
        return self.encoder(src, src_key_padding_mask=mask, pos=pos)


class TransformerEncoder(nn.Module):
    """Stack of transformer encoder layers."""
    
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Create a ModuleList of encoder layers
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
    
    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        Pass the input through each encoder layer in turn.
        
        Args:
            src: Input tensor
            src_mask: Optional mask for src
            src_key_padding_mask: Optional key padding mask
            pos: Optional position encoding
        
        Returns:
            Output tensor after all encoder layers
        """
        output = src
        
        for layer in self.layers:
            output = layer(output, src_mask=src_mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention and feedforward network."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        """Add position embeddings to the tensor if provided."""
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        """
        Forward pass through the encoder layer.
        
        Args:
            src: Input tensor of shape [seq_len, batch_size, d_model]
            src_mask: Optional mask for src
            src_key_padding_mask: Optional key padding mask (True for padding)
            pos: Optional position encoding
        
        Returns:
            Output tensor after self-attention and feedforward
        """
        # Add position embeddings to queries and keys
        q = k = self.with_pos_embed(src, pos)
        
        # Self-attention
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        
        # Add & norm (first block)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        
        # Add & norm (second block)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


def build_vl_transformer(args):
    """
    Build a visual-linguistic transformer from arguments.
    
    Args:
        args: Arguments containing transformer configuration
    
    Returns:
        VLTransformer instance
    """
    return VLTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        activation="relu",
    ) 