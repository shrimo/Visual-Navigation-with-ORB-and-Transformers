import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerNavigation(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes, dropout=0.3):
        """
        Transformer model for navigation task.
        Args:
            input_dim (int): Dimension of the one-hot input tokens.
            hidden_dim (int): Dimension of the hidden layers.
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads.
            num_classes (int): Number of output classes.
            dropout (float): Dropout probability (for regularization).
        """
        super(TransformerNavigation, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Set batch_first=True for better performance and to work with nested tensors
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        # Average pooling over sequence dimension (dim=1)
        x = x.mean(dim=1)
        out = self.fc(x)
        return out