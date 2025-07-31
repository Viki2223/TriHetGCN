import os
import time
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import random
from collections import defaultdict

from model.trihetgcn import TriHetGCN
from model.heuristics import evaluate_heuristic_model
from shared.utils import (
    save_model,
    plot_metrics,
    save_results_table,
    compare_with_paper,
    save_overall_results
)
from shared.graph_visualization import visualize_graph_structure

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ensure_feature_label_alignment(data):
    num_nodes = data.x.shape[0]
    if data.y.shape[0] > num_nodes:
        data.y = data.y[:num_nodes]
    elif data.y.shape[0] < num_nodes:
        pad = num_nodes - data.y.shape[0]
        data.y = torch.cat([data.y, torch.zeros(pad, dtype=torch.long)])
    return data

def get_dataset_hyperparameters(dataset_name):
    """Get optimized hyperparameters for each dataset and model"""
    params = {
        "Cora": {
            "GCN": {"lr": 0.01, "hidden_dim": 128, "epochs": 200, "dropout": 0.5},
            "GraphSAGE": {"lr": 0.01, "hidden_dim": 128, "epochs": 200, "dropout": 0.5},
            "GAT": {"lr": 0.005, "hidden_dim": 64, "epochs": 300, "dropout": 0.6},
            "TriHetGCN": {"lr": 0.01, "hidden_dim": 128, "epochs": 400, "dropout": 0.3}
        },
        "CiteSeer": {
            "GCN": {"lr": 0.01, "hidden_dim": 256, "epochs": 300, "dropout": 0.5},
            "GraphSAGE": {"lr": 0.01, "hidden_dim": 256, "epochs": 250, "dropout": 0.5},
            "GAT": {"lr": 0.005, "hidden_dim": 128, "epochs": 400, "dropout": 0.6},
            "TriHetGCN": {"lr": 0.01, "hidden_dim": 384, "epochs": 500, "dropout": 0.2}
        },
        "PubMed": {
            "GCN": {"lr": 0.01, "hidden_dim": 256, "epochs": 200, "dropout": 0.5},
            "GraphSAGE": {"lr": 0.005, "hidden_dim": 128, "epochs": 150, "dropout": 0.4},
            "GAT": {"lr": 0.005, "hidden_dim": 128, "epochs": 200, "dropout": 0.6},
            "TriHetGCN": {"lr": 0.005, "hidden_dim": 128, "epochs": 300, "dropout": 0.1}
        }
    }
    return params.get(dataset_name, {})

def run(dataset_name):
    print(f"{'=' * 65}\n🧠 Training on {dataset_name}\n{'=' * 65}")
    
    # Create results directory
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Created results directory: {results_dir}")
    
    dataset = Planetoid(root=f"./Datasets/{dataset_name}", name=dataset_name)
    data = ensure_feature_label_alignment(dataset[0])
    edge_index = data.edge_index

    print(f"⏬ Loaded {data.num_nodes} nodes, {edge_index.shape[1]} edges for {dataset_name}")

    # Visualize graph structure with hub nodes
    graph_img_path = f"{results_dir}/graph_{dataset_name}.png"
    visualize_graph_structure(edge_index, data.y, dataset_name, graph_img_path)

    # Get hyperparameters for this dataset
    hyperparams = get_dataset_hyperparameters(dataset_name)

    # Initialize results dictionary
    results = {}
    
    # Split edges using RandomLinkSplit with fixed seed for reproducibility
    print("✂️ Splitting edges with 85% train, 5% validation, 10% test...")
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    transform = RandomLinkSplit(
        num_val=0.05,
        num_test=0.10,
        is_undirected=True,
        add_negative_train_samples=False,
        neg_sampling_ratio=1.0
    )
    train_data, val_data, test_data = transform(data)
    
    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)
    test_data = test_data.to(device)
    
    # Extract test edges and labels
    test_pos = test_data.edge_label_index[:, test_data.edge_label == 1]
    test_neg = test_data.edge_label_index[:, test_data.edge_label == 0]
    
    # Heuristic evaluation with improved implementations
    print("🔬 Running heuristics...")
    for method in ["CN", "AA", "RA", "Katz"]:
        try:
            auc, ap = evaluate_heuristic_model(method, train_data, test_pos, test_neg)
            results[method] = (auc, ap)
            print(f"🎯 {method} AUC: {auc*100:.2f}%, AP: {ap*100:.2f}%")
        except Exception as e:
            print(f"❌ {method} failed: {e}")

    # Generate negative training edges with better sampling strategy
    num_train_pos = train_data.edge_index.size(1)
    num_train_neg = num_train_pos
    existing_edges_set = set()
    for u, v in train_data.edge_index.t().tolist():
        u, v = min(u, v), max(u, v)
        existing_edges_set.add((u, v))
    
    # Improved negative sampling with degree-based sampling
    train_neg = []
    all_nodes = list(range(train_data.num_nodes))
    degree_list = torch.zeros(train_data.num_nodes)
    for edge in train_data.edge_index.t():
        degree_list[edge[0]] += 1
        degree_list[edge[1]] += 1
    
    # Normalize degrees for sampling probability
    degree_prob = degree_list / degree_list.sum()
    
    while len(train_neg) < num_train_neg:
        # Sample nodes based on degree distribution
        if random.random() < 0.7:  # 70% degree-based sampling
            u = np.random.choice(all_nodes, p=degree_prob.numpy())
            v = np.random.choice(all_nodes, p=degree_prob.numpy())
        else:  # 30% uniform sampling
            u = random.choice(all_nodes)
            v = random.choice(all_nodes)
            
        if u == v:
            continue
        u, v = min(u, v), max(u, v)
        if (u, v) in existing_edges_set:
            continue
        train_neg.append([u, v])
        existing_edges_set.add((u, v))
    
    train_neg_tensor = torch.tensor(train_neg, dtype=torch.long).t().to(device)
    train_edge_index = torch.cat([train_data.edge_index, train_neg_tensor], dim=1)
    train_labels = torch.cat([
        torch.ones(num_train_pos, device=device),
        torch.zeros(num_train_neg, device=device)
    ])

    # Train models with optimized hyperparameters
    for model_name in ["GCN", "GraphSAGE", "GAT", "TriHetGCN"]:
        print(f"\n📈 Training {model_name} - {dataset_name}")
        
        # Get model-specific hyperparameters
        model_params = hyperparams.get(model_name, {
            "lr": 0.01, "hidden_dim": 128, "epochs": 200, "dropout": 0.5
        })
        
        # Enhanced model initialization
        model = TriHetGCN(
            model=model_name, 
            in_channels=data.x.shape[1], 
            hidden_channels=model_params["hidden_dim"],
            out_channels=model_params["hidden_dim"],
            num_anchors=100 if model_name == "TriHetGCN" else 0,
            dropout=model_params["dropout"]
        ).to(device)
        
        data_for_gnn = train_data
        optimizer = torch.optim.Adam(model.parameters(), lr=model_params["lr"], weight_decay=5e-4)
        criterion = torch.nn.BCELoss()
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=50, verbose=False
        )
        
        losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = 100
        
        num_epochs = model_params["epochs"]
            
        model.train()
        for epoch in tqdm(range(num_epochs), desc=model_name, mininterval=1):
            optimizer.zero_grad()
            
            # Forward pass with edge-based scoring
            preds = model(data_for_gnn, train_edge_index)
            loss = criterion(preds, train_labels)
            
            # Add L2 regularization
            l2_reg = 0
            for param in model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += 1e-5 * l2_reg
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            losses.append(loss.item())
            
            # Validation and early stopping
            if epoch % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_edges = val_data.edge_label_index
                    val_preds = model(data_for_gnn, val_edges)
                    val_loss = criterion(val_preds, val_data.edge_label.float())
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        best_model_state = model.state_dict().copy()
                    else:
                        patience_counter += 1
                    
                    scheduler.step(val_loss)
                    
                model.train()
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Memory management for large datasets
            if dataset_name == "PubMed" and epoch % 10 == 0:
                torch.cuda.empty_cache()

        # Load best model
        if 'best_model_state' in locals():
            model.load_state_dict(best_model_state)

        # Save loss curve
        loss_img_path = f"{results_dir}/{dataset_name}_loss_{model_name}.png"
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(losses)+1), losses, marker='o', linewidth=2, markersize=4)
        plt.title(f"{model_name} Training Loss - {dataset_name}", fontsize=14, fontweight='bold')
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(loss_img_path, dpi=300)
        plt.close()
        print(f"💾 Saved loss curve to {loss_img_path}")

        # Enhanced evaluation with test-time augmentation
        try:
            model.eval()
            with torch.no_grad():
                test_edges = test_data.edge_label_index
                y_label = test_data.edge_label.cpu().numpy()
                
                # Multiple forward passes for better stability (test-time augmentation)
                y_scores_list = []
                for _ in range(5 if model_name == "TriHetGCN" else 3):
                    if dataset_name == "PubMed":
                        # Batch processing for large datasets
                        batch_size = 8000
                        y_score = []
                        for i in range(0, test_edges.size(1), batch_size):
                            batch = test_edges[:, i:i+batch_size]
                            batch_scores = model(data_for_gnn, batch).cpu().numpy()
                            y_score.extend(batch_scores)
                        y_scores_list.append(np.array(y_score))
                    else:
                        y_score = model(data_for_gnn, test_edges).cpu().numpy()
                        y_scores_list.append(y_score)
                
                # Average predictions
                y_score = np.mean(y_scores_list, axis=0)
                    
            auc = roc_auc_score(y_label, y_score)
            ap = average_precision_score(y_label, y_score)
            results[model_name] = (auc, ap)

            print(f"🎯 {model_name} AUC: {auc*100:.2f}%, AP: {ap*100:.2f}%")
        except Exception as e:
            print(f"❌ AUC/AP computation failed for {model_name}: {e}")

        # Save model and clear memory
        checkpoint_dir = f"{results_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        save_path = f"{checkpoint_dir}/{dataset_name}_{model_name}.pt"
        save_model(model, save_path)
        print(f"💾 Saved model to {save_path}")
        del model
        if 'best_model_state' in locals():
            del best_model_state
        torch.cuda.empty_cache()

    # Print and save comprehensive results
    print("\n📊 Final Results Summary:")
    results_table = save_results_table(results, dataset_name, results_dir)
    print(results_table)
    
    # Plot metrics
    plot_metrics(results, dataset_name, results_dir)
    
    # Compare with paper results
    compare_with_paper(results, dataset_name, results_dir)
    
    print(f"✅ Done with {dataset_name}\n")

if __name__ == "__main__":
    # Set global seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    for ds in ["Cora", "CiteSeer", "PubMed"]:
        run(ds)
    
    # Generate overall summary
    save_overall_results()  