import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, BatchNorm
from torch_geometric.utils import dropout_adj, add_self_loops


class OptimizedGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=768, out_channels=384, dropout=0.15, heads=8, use_sigmoid=True):
        super().__init__()
        
        # Optimized architecture with more capacity
        self.conv1 = GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True, 
                            dropout=dropout, add_self_loops=True, bias=True)
        self.conv2 = GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, 
                            dropout=dropout, add_self_loops=True, bias=True)
        self.conv3 = GATConv(hidden_channels, hidden_channels // (heads//2), heads=heads//2, concat=True, 
                            dropout=dropout, add_self_loops=True, bias=True)
        self.conv4 = GATConv(hidden_channels, hidden_channels // 4, heads=4, concat=True, 
                            dropout=dropout, add_self_loops=True, bias=True)
        self.conv5 = GATConv(hidden_channels, out_channels, heads=1, concat=False, 
                            dropout=dropout, add_self_loops=True, bias=True)
        
        # Advanced normalization and regularization
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.bn4 = BatchNorm(hidden_channels)
        self.bn5 = BatchNorm(out_channels)
        
        # Residual connections
        self.residual1 = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()
        self.residual2 = nn.Identity()
        self.residual3 = nn.Identity()
        self.residual4 = Linear(hidden_channels, out_channels) if hidden_channels != out_channels else nn.Identity()
        
        # Enhanced link predictor with attention mechanism
        self.edge_attention = nn.MultiheadAttention(out_channels, num_heads=4, dropout=dropout, batch_first=True)
        
        self.link_predictor = nn.Sequential(
            Linear(out_channels * 3, hidden_channels),  # *3 for src, dst, and attention
            nn.GELU(),
            BatchNorm(hidden_channels),
            nn.Dropout(dropout),
            Linear(hidden_channels, hidden_channels // 2),
            nn.GELU(),
            BatchNorm(hidden_channels // 2),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, hidden_channels // 4),
            nn.GELU(),
            BatchNorm(hidden_channels // 4),
            nn.Dropout(dropout),
            Linear(hidden_channels // 4, hidden_channels // 8),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 8, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.use_sigmoid = use_sigmoid

    def forward(self, data, edge_index=None):
        x, full_edge_index = data.x, data.edge_index
        
        # Apply edge dropout for regularization during training
        if self.training:
            full_edge_index, _ = dropout_adj(full_edge_index, p=0.1, training=self.training)
        
        # Progressive feature extraction with residual connections
        identity1 = self.residual1(x)
        x1 = self.conv1(x, full_edge_index)
        x1 = self.bn1(x1)
        x1 = F.elu(x1)
        if x1.shape == identity1.shape:
            x1 = x1 + identity1
        x1 = self.dropout(x1)
        
        # Second layer with residual
        x2 = self.conv2(x1, full_edge_index)
        x2 = self.bn2(x2)
        x2 = F.elu(x2) + x1  # Residual connection
        x2 = self.dropout(x2)
        
        # Third layer with residual
        x3 = self.conv3(x2, full_edge_index)
        x3 = self.bn3(x3)
        x3 = F.elu(x3) + x2  # Residual connection
        x3 = self.dropout(x3)
        
        # Fourth layer with residual
        x4 = self.conv4(x3, full_edge_index)
        x4 = self.bn4(x4)
        x4 = F.elu(x4) + x3  # Residual connection
        x4 = self.dropout(x4)
        
        # Final layer with skip connection
        identity4 = self.residual4(x1) if not isinstance(self.residual4, nn.Identity) else x1
        x5 = self.conv5(x4, full_edge_index)
        x5 = self.bn5(x5)
        if x5.shape == identity4.shape:
            x5 = x5 + identity4
        
        # Enhanced link prediction with attention
        if edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            
            # Get embeddings
            src_emb = x5[src]
            dst_emb = x5[dst]
            
            # Apply attention mechanism
            edge_pairs = torch.stack([src_emb, dst_emb], dim=1)  # [num_edges, 2, out_channels]
            attended_edges, _ = self.edge_attention(edge_pairs, edge_pairs, edge_pairs)
            attended_feature = attended_edges.mean(dim=1)  # [num_edges, out_channels]
            
            # Combine original and attended features
            edge_emb = torch.cat([src_emb, dst_emb, attended_feature], dim=1)
            
            # Predict link probability
            scores = self.link_predictor(edge_emb).squeeze()
            
            return torch.sigmoid(scores) if self.use_sigmoid else scores
        
        return x5