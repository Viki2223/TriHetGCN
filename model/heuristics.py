import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy.sparse as sp
import gc
import psutil

def get_memory_usage():
    """Get current memory usage in GB"""
    try:
        return psutil.Process().memory_info().rss / 1024 / 1024 / 1024
    except:
        return 0

def katz_matrix_optimized(adj, beta=0.005, max_iter=5, memory_limit_gb=12):
    """Improved Katz computation with better memory management and higher iterations"""
    try:
        n = adj.shape[0]
        print(f"🔍 Computing Katz for {n} nodes (Memory: {get_memory_usage():.2f}GB)")
        
        # Increase memory limits and node limits for better results
        if n > 25000 or get_memory_usage() > memory_limit_gb:
            print(f"⚠️ Skipping Katz for very large graph (n={n}) or low memory")
            return sp.csr_matrix((n, n))
        
        # Ensure we have CSR format for efficient operations
        if sp.issparse(adj):
            if not sp.isspmatrix_csr(adj):
                adj_sparse = adj.tocsr()
            else:
                adj_sparse = adj.copy()
        else:
            adj_sparse = sp.csr_matrix(adj)
        
        # Normalize the adjacency matrix to prevent overflow
        adj_sparse.data = adj_sparse.data.astype(np.float64)
        
        # Check if adjacency matrix is valid
        if adj_sparse.nnz == 0:
            print("⚠️ Empty adjacency matrix, returning zero Katz matrix")
            return sp.csr_matrix((n, n), dtype=np.float64)
        
        print(f"  Adjacency matrix: {n}x{n}, {adj_sparse.nnz} edges, beta={beta}")
        
        # Use power series: beta*A + (beta*A)^2 + (beta*A)^3 + ...
        katz_sparse = sp.csr_matrix((n, n), dtype=np.float64)
        A_power = (beta * adj_sparse).astype(np.float64)
        
        for i in range(1, max_iter + 1):
            print(f"  Katz iteration {i}/{max_iter}, adding {A_power.nnz} entries")
            katz_sparse = katz_sparse + A_power
            
            if i < max_iter:
                # A_power = beta * (adj_sparse @ A_power)
                A_power = (beta * (adj_sparse @ A_power)).astype(np.float64)
            
            # Memory and overflow checks
            if get_memory_usage() > memory_limit_gb:
                print(f"⚠️ Memory limit reached at iteration {i}")
                break
                
            # Check for numerical issues
            if np.any(np.isinf(A_power.data)) or np.any(np.isnan(A_power.data)):
                print(f"⚠️ Numerical issues detected at iteration {i}")
                break
                
            # Check if A_power becomes empty (convergence)
            if A_power.nnz == 0:
                print(f"  Convergence reached at iteration {i}")
                break
        
        # Remove diagonal elements (self-loops)
        katz_sparse.setdiag(0)
        
        # Clean up numerical issues
        katz_sparse.eliminate_zeros()
        
        print(f"✅ Katz computation completed. Non-zero entries: {katz_sparse.nnz}")
        print(f"  Katz matrix stats: min={katz_sparse.data.min():.8f}, max={katz_sparse.data.max():.8f}")
        
        return katz_sparse
        
    except Exception as e:
        print(f"❌ Katz computation failed: {str(e)}")
        return sp.csr_matrix((adj.shape[0], adj.shape[0]))

def heuristic_score_optimized(method, adj, src, dst, deg=None, katz_cache=None):
    """Improved heuristic scoring with better numerical stability"""
    try:
        if method == 'CN':
            # Common Neighbors
            if sp.issparse(adj):
                if not sp.isspmatrix_csr(adj):
                    adj = adj.tocsr()
                
                # Get neighbors for both nodes
                src_neighbors = set(adj[src].nonzero()[1])
                dst_neighbors = set(adj[dst].nonzero()[1])
                
                # Find intersection (common neighbors)
                common = src_neighbors.intersection(dst_neighbors)
                return len(common)
            else:
                # Dense matrix case
                common = np.logical_and(adj[src] > 0, adj[dst] > 0)
                return int(np.sum(common))
        
        elif method == 'AA':
            # Adamic-Adar Index
            if deg is None:
                if sp.issparse(adj):
                    deg = np.array(adj.sum(axis=1)).flatten()
                else:
                    deg = adj.sum(axis=1)
            
            if sp.issparse(adj):
                if not sp.isspmatrix_csr(adj):
                    adj = adj.tocsr()
                
                src_neighbors = set(adj[src].nonzero()[1])
                dst_neighbors = set(adj[dst].nonzero()[1])
                common_neighbors = src_neighbors.intersection(dst_neighbors)
            else:
                common = np.logical_and(adj[src] > 0, adj[dst] > 0)
                common_neighbors = set(np.where(common)[0])
            
            score = 0.0
            for neighbor in common_neighbors:
                degree = deg[neighbor]
                if degree > 1:  # Avoid log(1) = 0
                    score += 1.0 / np.log(degree)
            return score
        
        elif method == 'RA':
            # Resource Allocation Index
            if deg is None:
                if sp.issparse(adj):
                    deg = np.array(adj.sum(axis=1)).flatten()
                else:
                    deg = adj.sum(axis=1)
            
            if sp.issparse(adj):
                if not sp.isspmatrix_csr(adj):
                    adj = adj.tocsr()
                
                src_neighbors = set(adj[src].nonzero()[1])
                dst_neighbors = set(adj[dst].nonzero()[1])
                common_neighbors = src_neighbors.intersection(dst_neighbors)
            else:
                common = np.logical_and(adj[src] > 0, adj[dst] > 0)
                common_neighbors = set(np.where(common)[0])
            
            score = 0.0
            for neighbor in common_neighbors:
                degree = deg[neighbor]
                if degree > 0:
                    score += 1.0 / degree
            return score
        
        elif method == 'Katz':
            if sp.issparse(katz_cache):
                try:
                    # Direct sparse matrix indexing returns a scalar
                    value = katz_cache[src, dst]
                    return float(value)
                except Exception as e:
                    return 0.0
            elif katz_cache is not None:
                return float(katz_cache[src, dst])
            else:
                return 0.0
        
        else:
            raise ValueError(f"Unsupported method: {method}")
            
    except Exception as e:
        print(f"❌ Error in {method} for pair ({src}, {dst}): {e}")
        return 0.0

def evaluate_heuristic_model(method, data, pos_edge_index, neg_edge_index):
    """Improved evaluation with dataset-specific parameters"""
    print(f"\n🔍 Evaluating heuristic method: {method}")
    
    num_nodes = data.num_nodes
    print(f"📊 Graph size: {num_nodes} nodes, Memory: {get_memory_usage():.2f}GB")
    
    # Create adjacency matrix
    adj_sparse = to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    
    # Convert to CSR for efficient operations
    if not sp.isspmatrix_csr(adj_sparse):
        adj_sparse = adj_sparse.tocsr()
    
    # Binarize the adjacency matrix (remove weights, keep only 0/1)
    adj_sparse.data = (adj_sparse.data > 0).astype(np.float64)
    adj_sparse.eliminate_zeros()
    
    # Make the adjacency matrix symmetric for undirected graphs
    adj_sparse = adj_sparse + adj_sparse.T
    adj_sparse.data = (adj_sparse.data > 0).astype(np.float64)
    adj_sparse.eliminate_zeros()
    
    # Use sparse matrices for all computations to save memory
    adj = adj_sparse
    deg = np.array(adj.sum(axis=1)).flatten()
    
    # Precompute Katz if needed with dataset-specific parameters
    katz_cache = None
    if method == 'Katz':
        # Dataset-specific beta values based on empirical results
        dataset_betas = {
            'cora': 0.005,
            'citeseer': 0.005, 
            'pubmed': 0.001,    # Lower beta for PubMed
            'dblp': 0.001,
            'cs': 0.0005,
            'facebook': 0.0001,
            'power': 0.005,
            'twitter': 0.0001,
            'int': 0.001
        }
        
        # Infer dataset from graph characteristics
        beta = 0.005  # default
        if hasattr(data, 'name'):
            dataset_name = data.name.lower()
            beta = dataset_betas.get(dataset_name, beta)
        else:
            # Infer from size and structure
            if num_nodes > 50000:
                beta = 0.0001
            elif num_nodes > 20000:
                beta = 0.0005
            elif num_nodes > 15000:  # PubMed range
                beta = 0.001
            elif num_nodes > 10000:
                beta = 0.001
        
        print(f"Using beta = {beta} for Katz computation")
        
        # Additional safety check for beta value
        if sp.issparse(adj):
            max_degree = np.max(deg)
        else:
            max_degree = np.max(deg)
        
        # Beta should be less than 1/max_degree for convergence
        safe_beta = min(beta, 0.9 / max_degree) if max_degree > 0 else beta
        if safe_beta != beta:
            print(f"⚠️ Adjusting beta from {beta} to {safe_beta} for numerical stability (max_degree={max_degree})")
            beta = safe_beta
        
        katz_cache = katz_matrix_optimized(adj, beta=beta, max_iter=5)
    
    # Convert edge indices to CPU and numpy
    if isinstance(pos_edge_index, torch.Tensor):
        pos_edge_index = pos_edge_index.cpu().numpy()
    if isinstance(neg_edge_index, torch.Tensor):
        neg_edge_index = neg_edge_index.cpu().numpy()
    
    # Prepare edge pairs
    pos_pairs = list(zip(pos_edge_index[0], pos_edge_index[1]))
    neg_pairs = list(zip(neg_edge_index[0], neg_edge_index[1]))
    
    pos_scores = []
    neg_scores = []
    
    print(f"📊 Processing {len(pos_pairs)} positive edges...")
    batch_size = 1000  # Process in batches for better memory management
    
    # Evaluate positive edges in batches
    for i in tqdm(range(0, len(pos_pairs), batch_size), desc="Positive edges"):
        batch_pairs = pos_pairs[i:i+batch_size]
        batch_scores = []
        for src, dst in batch_pairs:
            score = heuristic_score_optimized(method, adj, src, dst, deg, katz_cache)
            batch_scores.append(score)
        pos_scores.extend(batch_scores)
    
    # Evaluate negative edges in batches
    print(f"📊 Processing {len(neg_pairs)} negative edges...")
    for i in tqdm(range(0, len(neg_pairs), batch_size), desc="Negative edges"):
        batch_pairs = neg_pairs[i:i+batch_size]
        batch_scores = []
        for src, dst in batch_pairs:
            score = heuristic_score_optimized(method, adj, src, dst, deg, katz_cache)
            batch_scores.append(score)
        neg_scores.extend(batch_scores)
    
    # Clean up memory
    del adj, katz_cache
    gc.collect()
    
    # Compute metrics
    y_true = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    y_scores = np.array(pos_scores + neg_scores)
    
    # Handle edge cases and ensure proper data types
    y_scores = np.nan_to_num(y_scores, nan=0.0, posinf=np.max(y_scores[np.isfinite(y_scores)]) if np.any(np.isfinite(y_scores)) else 1.0, neginf=0.0)
    
    # Debug information
    print(f"Score statistics: min={np.min(y_scores):.6f}, max={np.max(y_scores):.6f}, mean={np.mean(y_scores):.6f}, std={np.std(y_scores):.6f}")
    print(f"Positive scores: min={np.min(pos_scores):.6f}, max={np.max(pos_scores):.6f}, mean={np.mean(pos_scores):.6f}")
    print(f"Negative scores: min={np.min(neg_scores):.6f}, max={np.max(neg_scores):.6f}, mean={np.mean(neg_scores):.6f}")
    
    # Ensure we have variation in scores for proper ranking
    if np.std(y_scores) == 0:
        print("⚠️ All scores are identical, this indicates a problem with the computation")
        if method == 'Katz':
            print("⚠️ Katz scores are all zero - computation may have failed")
        # Add minimal noise as last resort
        y_scores += np.random.normal(0, 1e-10, len(y_scores))
    
    try:
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except Exception as e:
        print(f"❌ Metric computation failed: {e}")
        auc, ap = 0.5, np.mean(y_true)
    
    print(f"🎯 {method} - AUC: {auc*100:.2f}%, AP: {ap*100:.2f}%")
    return auc, ap

def evaluate_heuristic_model_fixed(method, data, pos_edge_index, neg_edge_index):
    """Legacy function name for backward compatibility"""
    return evaluate_heuristic_model(method, data, pos_edge_index, neg_edge_index)

# Additional helper function to run all heuristics at once
def evaluate_all_heuristics(data, pos_edge_index, neg_edge_index):
    """Evaluate all heuristic methods and return results in table format"""
    methods = ['CN', 'AA', 'RA', 'Katz']
    results = {}
    
    for method in methods:
        try:
            auc, ap = evaluate_heuristic_model(method, data, pos_edge_index, neg_edge_index)
            results[method] = {'AUC': auc * 100, 'AP': ap * 100}
        except Exception as e:
            print(f"❌ Failed to evaluate {method}: {e}")
            results[method] = {'AUC': 0.0, 'AP': 0.0}
    
    return results