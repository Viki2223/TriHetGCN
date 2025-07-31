import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, out_channels=256, dropout=0.4, use_sigmoid=True):
        super().__init__()
        
        # Deeper architecture for better representation learning
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, out_channels)
        
        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.ln4 = nn.LayerNorm(out_channels)
        
        # Residual connections
        self.residual1 = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()
        self.residual2 = nn.Identity()
        self.residual3 = Linear(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()
        
        # Advanced link predictor
        self.link_predictor = nn.Sequential(
            Linear(out_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels // 2),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels // 4),
            nn.Dropout(dropout),
            Linear(hidden_channels // 4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.use_sigmoid = use_sigmoid

    def forward(self, data, edge_index=None):
        x, full_edge_index = data.x, data.edge_index
        
        # First GCN layer with residual
        identity1 = self.residual1(x)
        x1 = self.conv1(x, full_edge_index)
        x1 = self.ln1(x1)
        x1 = F.relu(x1)
        if x1.shape == identity1.shape:
            x1 = x1 + identity1
        x1 = self.dropout(x1)
        
        # Second GCN layer with residual
        x2 = self.conv2(x1, full_edge_index)
        x2 = self.ln2(x2)
        x2 = F.relu(x2) + x1  # Residual connection
        x2 = self.dropout(x2)
        
        # Third GCN layer with residual
        x3 = self.conv3(x2, full_edge_index)
        x3 = self.ln3(x3)
        x3 = F.relu(x3) + x2  # Residual connection
        x3 = self.dropout(x3)
        
        # Fourth GCN layer with residual
        identity3 = self.residual3(x1) if not isinstance(self.residual3, nn.Identity) else x1
        x4 = self.conv4(x3, full_edge_index)
        x4 = self.ln4(x4)
        if x4.shape == identity3.shape:
            x4 = x4 + identity3
        
        # Link prediction
        if edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            
            # Multi-scale edge features
            src_emb = x4[src]
            dst_emb = x4[dst]
            
            # Concatenate embeddings
            edge_emb = torch.cat([src_emb, dst_emb], dim=1)
            
            # Predict link probability
            scores = self.link_predictor(edge_emb).squeeze()
            
            return torch.sigmoid(scores) if self.use_sigmoid else scores
        
        return x4