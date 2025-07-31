import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear, BatchNorm
from torch_geometric.utils import dropout_adj


class OptimizedGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels=768, out_channels=384, dropout=0.15, use_sigmoid=True):
        super().__init__()
        
        # Multi-layer SAGE with diverse aggregations and increased capacity
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean', bias=True, normalize=True)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='max', bias=True, normalize=True)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean', bias=True, normalize=True)
        self.conv4 = SAGEConv(hidden_channels, hidden_channels // 2, aggr='max', bias=True, normalize=True)
        self.conv5 = SAGEConv(hidden_channels // 2, out_channels, aggr='mean', bias=True, normalize=True)
        
        # Advanced normalization
        self.bn1 = BatchNorm(hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)
        self.bn3 = BatchNorm(hidden_channels)
        self.bn4 = BatchNorm(hidden_channels // 2)
        self.bn5 = BatchNorm(out_channels)
        
        # Skip connections with proper dimension handling
        self.skip1 = Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()
        self.skip2 = nn.Identity()
        self.skip3 = nn.Identity()
        self.skip4 = Linear(hidden_channels, hidden_channels // 2) if hidden_channels != hidden_channels // 2 else nn.Identity()
        self.skip5 = Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
        # Multi-scale aggregation module
        self.multi_scale_conv1 = SAGEConv(hidden_channels, hidden_channels // 4, aggr='add', bias=True)
        self.multi_scale_conv2 = SAGEConv(hidden_channels, hidden_channels // 4, aggr='std', bias=True)
        
        # Enhanced link predictor with multi-head attention
        self.attention = nn.MultiheadAttention(out_channels, num_heads=8, dropout=dropout, batch_first=True)
        
        self.link_predictor = nn.Sequential(
            Linear(out_channels * 4, hidden_channels),  # Increased input due to multi-scale features
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
        
        # Auxiliary predictor for ensemble effect
        self.aux_predictor = nn.Sequential(
            Linear(out_channels, hidden_channels // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.use_sigmoid = use_sigmoid

    def forward(self, data, edge_index=None):
        x, full_edge_index = data.x, data.edge_index
        x_original = x.clone()
        
        # Apply edge dropout for regularization
        if self.training:
            full_edge_index, _ = dropout_adj(full_edge_index, p=0.1, training=self.training)
        
        # Progressive feature extraction with skip connections
        identity1 = self.skip1(x)
        x1 = self.conv1(x, full_edge_index)
        x1 = self.bn1(x1)
        x1 = F.gelu(x1)
        if x1.shape == identity1.shape:
            x1 = x1 + identity1
        x1 = self.dropout(x1)
        
        # Second layer with residual
        x2 = self.conv2(x1, full_edge_index)
        x2 = self.bn2(x2)
        x2 = F.gelu(x2) + x1  # Residual connection
        x2 = self.dropout(x2)
        
        # Multi-scale aggregation at intermediate layer
        ms1 = self.multi_scale_conv1(x2, full_edge_index)
        ms2 = self.multi_scale_conv2(x2, full_edge_index)
        x2_enhanced = torch.cat([x2, ms1, ms2], dim=1)
        
        # Dimension reduction for consistency
        x2_proj = Linear(x2_enhanced.shape[1], x2.shape[1]).to(x2.device)(x2_enhanced)
        x2 = x2 + x2_proj
        
        # Third layer with residual
        x3 = self.conv3(x2, full_edge_index)
        x3 = self.bn3(x3)
        x3 = F.gelu(x3) + x2  # Residual connection
        x3 = self.dropout(x3)
        
        # Fourth layer with skip connection
        identity4 = self.skip4(x2) if not isinstance(self.skip4, nn.Identity) else x2
        x4 = self.conv4(x3, full_edge_index)
        x4 = self.bn4(x4)
        x4 = F.gelu(x4)
        if x4.shape == identity4.shape:
            x4 = x4 + identity4
        x4 = self.dropout(x4)
        
        # Final layer with long skip connection
        identity5 = self.skip5(x_original) if not isinstance(self.skip5, nn.Identity) else x_original
        x5 = self.conv5(x4, full_edge_index)
        x5 = self.bn5(x5)
        if x5.shape == identity5.shape:
            x5 = x5 + identity5
        
        # Enhanced link prediction
        if edge_index is not None:
            src, dst = edge_index[0], edge_index[1]
            
            # Get embeddings
            src_emb = x5[src]
            dst_emb = x5[dst]
            
            # Apply attention mechanism
            edge_pairs = torch.stack([src_emb, dst_emb], dim=1)
            attended_edges, attention_weights = self.attention(edge_pairs, edge_pairs, edge_pairs)
            attended_feature = attended_edges.mean(dim=1)
            
            # Multi-scale edge features
            edge_concat = torch.cat([src_emb, dst_emb], dim=1)
            edge_hadamard = src_emb * dst_emb  # Element-wise multiplication
            edge_diff = torch.abs(src_emb - dst_emb)  # Absolute difference
            
            # Combine all features
            combined_features = torch.cat([edge_concat, edge_hadamard, edge_diff, attended_feature], dim=1)
            
            # Main prediction
            scores_main = self.link_predictor(combined_features).squeeze()
            
            # Auxiliary prediction for ensemble
            scores_aux = self.aux_predictor(attended_feature).squeeze()
            
            # Weighted combination
            combined_scores = 0.8 * scores_main + 0.2 * scores_aux
            
            return torch.sigmoid(combined_scores) if self.use_sigmoid else combined_scores
        
        return x5