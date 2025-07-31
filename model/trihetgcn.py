import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.utils import degree, negative_sampling
from torch_geometric.data import Data
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import networkx as nx
from collections import deque
import random
import time


class TriHetGCN(nn.Module):
    def __init__(self, model, in_channels, hidden_channels=512, out_channels=256, 
                 num_anchors=30, dropout=0.4, use_sigmoid=True):
        super().__init__()
        self.model_name = model
        self.num_anchors = num_anchors
        self.dropout = dropout
        self.use_sigmoid = use_sigmoid
        
        # Cache for computed features
        self._feature_cache = {}
        
        if model == "TriHetGCN":
            # Enhanced feature engineering with more comprehensive features
            topo_dim = num_anchors
            struct_dim = 12  # Increased structural features
            enhanced_dim = in_channels + topo_dim + struct_dim
            
            # Feature preprocessing
            self.feature_preprocessor = nn.Sequential(
                Linear(enhanced_dim, enhanced_dim),
                nn.LayerNorm(enhanced_dim),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5)
            )
            
            # Deeper GCN architecture for TriHetGCN
            self.conv1 = GCNConv(enhanced_dim, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.conv3 = GCNConv(hidden_channels, hidden_channels)
            self.conv4 = GCNConv(hidden_channels, out_channels)
            
        else:
            # Optimized standard models with deeper architectures
            if model == "GCN":
                self.conv1 = GCNConv(in_channels, hidden_channels)
                self.conv2 = GCNConv(hidden_channels, hidden_channels)
                self.conv3 = GCNConv(hidden_channels, hidden_channels)
                self.conv4 = GCNConv(hidden_channels, out_channels)
            elif model == "GraphSAGE":
                self.conv1 = SAGEConv(in_channels, hidden_channels, aggr='mean')
                self.conv2 = SAGEConv(hidden_channels, hidden_channels, aggr='max')
                self.conv3 = SAGEConv(hidden_channels, hidden_channels, aggr='mean')
                self.conv4 = SAGEConv(hidden_channels, out_channels, aggr='max')
            elif model == "GAT":
                heads = 8
                self.conv1 = GATConv(in_channels, hidden_channels//heads, heads=heads, concat=True, dropout=dropout)
                self.conv2 = GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True, dropout=dropout)
                self.conv3 = GATConv(hidden_channels, hidden_channels//heads, heads=heads, concat=True, dropout=dropout)
                self.conv4 = GATConv(hidden_channels, out_channels, heads=1, concat=False, dropout=dropout)
        
        # Layer normalization for all models
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        self.ln3 = nn.LayerNorm(hidden_channels)
        self.ln4 = nn.LayerNorm(out_channels)
        
        # Advanced link predictor with attention mechanism
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
        
        self.dropout_layer = nn.Dropout(dropout)

    def _efficient_bfs_distances(self, adj_list, start_node, max_nodes, max_distance=4):
        """Efficient BFS for distance computation"""
        distances = [-1] * max_nodes
        distances[start_node] = 0
        queue = deque([start_node])
        
        while queue:
            node = queue.popleft()
            if distances[node] >= max_distance:
                continue
                
            for neighbor in adj_list[node]:
                if distances[neighbor] == -1:
                    distances[neighbor] = distances[node] + 1
                    queue.append(neighbor)
        
        # Convert -1 to max_distance + 1 for unreachable nodes
        distances = [d if d != -1 else max_distance + 1 for d in distances]
        return distances

    def _compute_topology_features(self, edge_index, num_nodes):
        """Compute advanced topology features with caching"""
        cache_key = f"topo_{num_nodes}_{edge_index.shape[1]}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        device = edge_index.device
        
        # Compute node degrees
        degrees = degree(edge_index[0], num_nodes=num_nodes)
        
        # Select diverse anchors: high degree, medium degree, low degree, random
        sorted_degrees, sorted_indices = torch.sort(degrees, descending=True)
        
        anchor_indices = []
        
        # High degree nodes
        high_degree_count = min(self.num_anchors // 3, num_nodes // 10)
        anchor_indices.extend(sorted_indices[:high_degree_count].tolist())
        
        # Medium degree nodes
        medium_start = num_nodes // 4
        medium_end = 3 * num_nodes // 4
        medium_degree_count = min(self.num_anchors // 3, medium_end - medium_start)
        if medium_start < len(sorted_indices):
            anchor_indices.extend(sorted_indices[medium_start:medium_start + medium_degree_count].tolist())
        
        # Low degree nodes
        low_degree_count = min(self.num_anchors // 3, num_nodes // 10)
        anchor_indices.extend(sorted_indices[-low_degree_count:].tolist())
        
        # Fill remaining with random nodes
        remaining_count = self.num_anchors - len(anchor_indices)
        if remaining_count > 0:
            all_nodes = set(range(num_nodes))
            used_nodes = set(anchor_indices)
            available_nodes = list(all_nodes - used_nodes)
            if available_nodes:
                random_nodes = np.random.choice(available_nodes, 
                                             min(remaining_count, len(available_nodes)), 
                                             replace=False)
                anchor_indices.extend(random_nodes.tolist())
        
        # Ensure we have exactly num_anchors
        anchor_indices = anchor_indices[:self.num_anchors]
        
        # Build adjacency list for BFS
        adj_list = [[] for _ in range(num_nodes)]
        edge_list = edge_index.t().cpu().numpy()
        for src, dst in edge_list:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        # Compute distances to anchors
        topo_features = torch.zeros((num_nodes, self.num_anchors), device=device)
        
        for i, anchor_idx in enumerate(anchor_indices):
            distances = self._efficient_bfs_distances(adj_list, anchor_idx, num_nodes)
            distances_tensor = torch.tensor(distances, device=device, dtype=torch.float)
            
            # Convert distances to similarities
            max_dist = distances_tensor.max()
            if max_dist > 0:
                similarities = torch.exp(-distances_tensor / max_dist)
            else:
                similarities = torch.ones_like(distances_tensor)
            
            topo_features[:, i] = similarities
        
        self._feature_cache[cache_key] = topo_features
        return topo_features

    def _compute_structural_features(self, edge_index, num_nodes):
        """Compute comprehensive structural features with error handling"""
        cache_key = f"struct_{num_nodes}_{edge_index.shape[1]}"
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        device = edge_index.device
        
        # Basic degree features
        degrees = degree(edge_index[0], num_nodes=num_nodes).float()
        degree_centrality = degrees / (degrees.max() + 1e-8)
        
        # Convert to NetworkX for advanced features
        edge_list = edge_index.t().cpu().numpy()
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        G.add_edges_from(edge_list)
        
        # Compute centralities with fallbacks
        try:
            # Use a sample for large graphs to speed up computation
            sample_size = min(1000, num_nodes)
            sample_nodes = np.random.choice(num_nodes, sample_size, replace=False) if num_nodes > sample_size else None
            
            closeness_dict = nx.closeness_centrality(G, u=sample_nodes)
            closeness = torch.zeros(num_nodes, device=device)
            for node, value in closeness_dict.items():
                closeness[node] = value
        except:
            closeness = degree_centrality.clone()
        
        try:
            betweenness_dict = nx.betweenness_centrality(G, k=min(100, num_nodes))
            betweenness = torch.zeros(num_nodes, device=device)
            for node, value in betweenness_dict.items():
                betweenness[node] = value
        except:
            betweenness = degree_centrality.clone()
        
        try:
            pagerank_dict = nx.pagerank(G, max_iter=100, tol=1e-4)
            pagerank = torch.tensor([pagerank_dict[i] for i in range(num_nodes)], device=device)
        except:
            pagerank = degree_centrality.clone()
        
        try:
            clustering_dict = nx.clustering(G)
            clustering = torch.tensor([clustering_dict[i] for i in range(num_nodes)], device=device)
        except:
            clustering = torch.zeros_like(degree_centrality)
        
        # Additional structural features
        degree_squared = degrees ** 2
        degree_log = torch.log(degrees + 1)
        degree_sqrt = torch.sqrt(degrees)
        triangles = clustering * degrees * (degrees - 1) / 2
        
        # Local clustering coefficient variations
        local_clustering = clustering
        avg_neighbor_degree = torch.zeros_like(degrees)
        
        # Compute average neighbor degree
        adj_list = [[] for _ in range(num_nodes)]
        for src, dst in edge_list:
            adj_list[src].append(dst)
            adj_list[dst].append(src)
        
        for node in range(num_nodes):
            if len(adj_list[node]) > 0:
                neighbor_degrees = [degrees[neighbor].item() for neighbor in adj_list[node]]
                avg_neighbor_degree[node] = sum(neighbor_degrees) / len(neighbor_degrees)
        
        # Normalize features
        def safe_normalize(tensor):
            max_val = tensor.max()
            return tensor / (max_val + 1e-8) if max_val > 0 else tensor
        
        structural_features = torch.stack([
            degree_centrality,
            closeness,
            betweenness,
            pagerank,
            clustering,
            safe_normalize(degree_squared),
            safe_normalize(degree_log),
            safe_normalize(degree_sqrt),
            safe_normalize(triangles),
            local_clustering,
            safe_normalize(avg_neighbor_degree),
            safe_normalize(degrees * pagerank)  # Combined feature
        ], dim=1)
        
        self._feature_cache[cache_key] = structural_features
        return structural_features

    def forward(self, data, edge_index=None):
        x, full_edge_index = data.x, data.edge_index
        num_nodes = x.size(0)
        
        if self.model_name == "TriHetGCN":
            # Enhanced feature engineering
            topo_features = self._compute_topology_features(full_edge_index, num_nodes)
            struct_features = self._compute_structural_features(full_edge_index, num_nodes)
            
            # Combine all features
            enhanced_x = torch.cat([x, topo_features, struct_features], dim=1)
            enhanced_x = self.feature_preprocessor(enhanced_x)
            x = enhanced_x
        
        # Four-layer architecture with residual connections
        x1 = self.conv1(x, full_edge_index)
        x1 = self.ln1(x1)
        if self.model_name == "GAT":
            x1 = F.elu(x1)
        else:
            x1 = F.relu(x1)
        x1 = self.dropout_layer(x1)
        
        x2 = self.conv2(x1, full_edge_index)
        x2 = self.ln2(x2)
        if self.model_name == "GAT":
            x2 = F.elu(x2) + x1  # Residual
        else:
            x2 = F.relu(x2) + x1  # Residual
        x2 = self.dropout_layer(x2)
        
        x3 = self.conv3(x2, full_edge_index)
        x3 = self.ln3(x3)
        if self.model_name == "GAT":
            x3 = F.elu(x3) + x2  # Residual
        else:
            x3 = F.relu(x3) + x2  # Residual
        x3 = self.dropout_layer(x3)
        
        x4 = self.conv4(x3, full_edge_index)
        x4 = self.ln4(x4)
        
        # Advanced link prediction
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

    def clear_cache(self):
        """Clear feature cache"""
        self._feature_cache.clear()


def compute_baseline_scores(edge_index, test_edges, method='CN'):
    """Optimized baseline computation"""
    device = edge_index.device
    num_nodes = edge_index.max().item() + 1
    
    # Create adjacency matrix
    adj_indices = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    adj_values = torch.ones(adj_indices.shape[1], device=device)
    adj = torch.sparse_coo_tensor(adj_indices, adj_values, (num_nodes, num_nodes)).coalesce()
    adj_dense = adj.to_dense()
    
    src, dst = test_edges[0], test_edges[1]
    scores = torch.zeros(len(src), device=device)
    
    if method == 'CN':  # Common Neighbors
        for i in range(len(src)):
            scores[i] = (adj_dense[src[i]] * adj_dense[dst[i]]).sum()
            
    elif method == 'AA':  # Adamic-Adar
        degrees = torch.sparse.sum(adj, dim=1).to_dense()
        for i in range(len(src)):
            common = adj_dense[src[i]] * adj_dense[dst[i]]
            common_neighbors = common.nonzero().squeeze(-1)
            if len(common_neighbors) > 0:
                neighbor_degrees = degrees[common_neighbors]
                valid_mask = neighbor_degrees > 1
                if valid_mask.sum() > 0:
                    scores[i] = (1.0 / torch.log(neighbor_degrees[valid_mask])).sum()
                    
    elif method == 'RA':  # Resource Allocation
        degrees = torch.sparse.sum(adj, dim=1).to_dense()
        for i in range(len(src)):
            common = adj_dense[src[i]] * adj_dense[dst[i]]
            common_neighbors = common.nonzero().squeeze(-1)
            if len(common_neighbors) > 0:
                neighbor_degrees = degrees[common_neighbors]
                valid_mask = neighbor_degrees > 0
                if valid_mask.sum() > 0:
                    scores[i] = (1.0 / neighbor_degrees[valid_mask]).sum()
    
    return scores


def create_train_val_test_split(edge_index, num_nodes, val_ratio=0.1, test_ratio=0.2):
    """Create train/validation/test splits with proper negative sampling"""
    device = edge_index.device
    
    # Convert to undirected and remove self-loops
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    edge_index = torch.unique(edge_index, dim=1)
    
    # Remove self-loops
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    
    num_edges = edge_index.shape[1]
    perm = torch.randperm(num_edges)
    
    # Split edges
    test_size = int(test_ratio * num_edges)
    val_size = int(val_ratio * num_edges)
    
    test_edges = edge_index[:, perm[:test_size]]
    val_edges = edge_index[:, perm[test_size:test_size + val_size]]
    train_edges = edge_index[:, perm[test_size + val_size:]]
    
    # Generate negative samples
    def generate_negative_edges(pos_edges, num_nodes, num_neg_edges):
        neg_edges = negative_sampling(
            edge_index=pos_edges,
            num_nodes=num_nodes,
            num_neg_samples=num_neg_edges,
            method='sparse'
        )
        return neg_edges
    
    train_neg_edges = generate_negative_edges(train_edges, num_nodes, train_edges.shape[1])
    val_neg_edges = generate_negative_edges(train_edges, num_nodes, val_edges.shape[1])
    test_neg_edges = generate_negative_edges(train_edges, num_nodes, test_edges.shape[1])
    
    return {
        'train_pos_edges': train_edges,
        'train_neg_edges': train_neg_edges,
        'val_pos_edges': val_edges,
        'val_neg_edges': val_neg_edges,
        'test_pos_edges': test_edges,
        'test_neg_edges': test_neg_edges
    }


def evaluate_model(model, data, edges_dict, device):
    """Comprehensive model evaluation"""
    model.eval()
    
    with torch.no_grad():
        # Get embeddings
        embeddings = model(data)
        
        results = {}
        
        for split in ['val', 'test']:
            pos_edges = edges_dict[f'{split}_pos_edges']
            neg_edges = edges_dict[f'{split}_neg_edges']
            
            # Positive scores
            pos_scores = model(data, pos_edges)
            neg_scores = model(data, neg_edges)
            
            # Combine scores and labels
            scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()
            labels = torch.cat([
                torch.ones(pos_edges.shape[1]),
                torch.zeros(neg_edges.shape[1])
            ]).numpy()
            
            # Compute metrics
            auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            
            results[f'{split}_auc'] = auc
            results[f'{split}_ap'] = ap
    
    return results


def train_model(model, data, edges_dict, device, epochs=500, lr=0.001, weight_decay=1e-5):
    """Advanced training loop with optimizations"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.8, patience=20, min_lr=1e-6
    )
    
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    patience_counter = 0
    patience = 50
    
    train_pos_edges = edges_dict['train_pos_edges']
    train_neg_edges = edges_dict['train_neg_edges']
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # Sample batch of edges for efficiency
        batch_size = min(4096, train_pos_edges.shape[1])
        pos_perm = torch.randperm(train_pos_edges.shape[1])[:batch_size]
        neg_perm = torch.randperm(train_neg_edges.shape[1])[:batch_size]
        
        batch_pos_edges = train_pos_edges[:, pos_perm]
        batch_neg_edges = train_neg_edges[:, neg_perm]
        
        # Forward pass
        pos_scores = model(data, batch_pos_edges)
        neg_scores = model(data, batch_neg_edges)
        
        # Compute loss
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([
            torch.ones(batch_pos_edges.shape[1], device=device),
            torch.zeros(batch_neg_edges.shape[1], device=device)
        ])
        
        loss = criterion(scores, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Evaluation
        if epoch % 10 == 0:
            results = evaluate_model(model, data, edges_dict, device)
            val_auc = results['val_auc']
            
            scheduler.step(val_auc)
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model state
                best_state = model.state_dict().copy()
            else:
                patience_counter += 1
            
            if epoch % 50 == 0:
                print(f'Epoch {epoch:3d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AP: {results["val_ap"]:.4f}')
        
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}')
            break
    
    # Load best model
    model.load_state_dict(best_state)
    return model


def run_comprehensive_experiment(data, device, models=['TriHetGCN', 'GCN', 'GraphSAGE', 'GAT']):
    """Run comprehensive experiments with all models"""
    
    print("Creating train/val/test splits...")
    edges_dict = create_train_val_test_split(data.edge_index, data.num_nodes)
    
    # Update data with training edges only
    train_data = Data(x=data.x, edge_index=edges_dict['train_pos_edges'], num_nodes=data.num_nodes)
    
    results = {}
    
    for model_name in models:
        print(f"\n{'='*50}")
        print(f"Training {model_name}")
        print(f"{'='*50}")
        
        # Initialize model
        if model_name == "TriHetGCN":
            model = TriHetGCN(
                model=model_name,
                in_channels=data.x.shape[1],
                hidden_channels=512,
                out_channels=256,
                num_anchors=50,  # Increased for better performance
                dropout=0.3,
                use_sigmoid=False  # Use logits for BCEWithLogitsLoss
            ).to(device)
        else:
            model = TriHetGCN(
                model=model_name,
                in_channels=data.x.shape[1],
                hidden_channels=512,
                out_channels=256,
                dropout=0.3,
                use_sigmoid=False
            ).to(device)
        
        # Train model
        start_time = time.time()
        model = train_model(model, train_data, edges_dict, device, epochs=500, lr=0.001)
        training_time = time.time() - start_time
        
        # Final evaluation
        final_results = evaluate_model(model, train_data, edges_dict, device)
        final_results['training_time'] = training_time
        
        results[model_name] = final_results
        
        print(f"\nFinal Results for {model_name}:")
        print(f"Test AUC: {final_results['test_auc']:.4f}")
        print(f"Test AP: {final_results['test_ap']:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        
        # Clear cache to save memory
        if hasattr(model, 'clear_cache'):
            model.clear_cache()
    
    # Print comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<15} {'Test AUC':<10} {'Test AP':<10} {'Time(s)':<10}")
    print("-" * 70)
    
    for model_name, res in results.items():
        print(f"{model_name:<15} {res['test_auc']:<10.4f} {res['test_ap']:<10.4f} {res['training_time']:<10.1f}")
    
    return results


# Example usage function
def create_sample_dataset(num_nodes=2000, num_features=64, avg_degree=10):
    """Create a sample dataset for testing"""
    
    # Generate random node features
    x = torch.randn(num_nodes, num_features)
    
    # Generate scale-free graph using preferential attachment
    G = nx.barabasi_albert_graph(num_nodes, avg_degree // 2)
    
    # Convert to edge_index format
    edge_list = list(G.edges())
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Create PyG data object
    data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
    
    return data


# Main execution example
if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create or load your dataset
    # Replace this with your actual dataset loading
    print("Creating sample dataset...")
    data = create_sample_dataset(num_nodes=3000, num_features=128, avg_degree=15)
    data = data.to(device)
    
    print(f"Dataset info:")
    print(f"- Nodes: {data.num_nodes}")
    print(f"- Edges: {data.edge_index.shape[1]}")
    print(f"- Features: {data.x.shape[1]}")
    
    # Run experiments
    results = run_comprehensive_experiment(data, device)
    
    # Find best performing model
    best_model = max(results.keys(), key=lambda k: results[k]['test_auc'])
    best_auc = results[best_model]['test_auc']
    
    print(f"\nBest performing model: {best_model}")
    print(f"Best Test AUC: {best_auc:.4f} ({best_auc*100:.1f}%)")
    
    if best_auc >= 0.90:
        print("🎉 Successfully achieved 90%+ performance!")
    else:
        print(f"Close to target! Try increasing epochs, adjusting hyperparameters, or using larger anchor sets.")