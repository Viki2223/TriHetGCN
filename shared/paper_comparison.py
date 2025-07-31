import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl

def compare_with_paper(results, dataset_name, results_dir):
    # Paper results from tables
    paper_results = {
        "Cora": {
            "AUC": {
                "CN": 71.94, "AA": 71.44, "RA": 71.90, "Katz": 83.76,
                "GCN": 92.73, "GraphSAGE": 92.21, "GAT": 92.14, "TriHetGCN": 93.69
            },
            "AP": {
                "CN": 71.72, "AA": 71.53, "RA": 72.00, "Katz": 84.32,
                "GCN": 93.51, "GraphSAGE": 93.09, "GAT": 93.13, "TriHetGCN": 94.40
            }
        },
        "CiteSeer": {
            "AUC": {
                "CN": 67.04, "AA": 66.86, "RA": 67.13, "Katz": 77.84,
                "GCN": 96.55, "GraphSAGE": 95.17, "GAT": 96.39, "TriHetGCN": 97.15
            },
            "AP": {
                "CN": 66.96, "AA": 66.88, "RA": 67.15, "Katz": 78.08,
                "GCN": 97.02, "GraphSAGE": 96.13, "GAT": 96.59, "TriHetGCN": 97.53
            }
        },
        "PubMed": {
            "AUC": {
                "CN": 64.39, "AA": 64.44, "RA": 64.42, "Katz": 81.91,
                "GCN": 97.18, "GraphSAGE": 94.32, "GAT": 95.64, "TriHetGCN": 97.21
            },
            "AP": {
                "CN": 64.36, "AA": 64.45, "RA": 64.43, "Katz": 82.71,
                "GCN": 97.18, "GraphSAGE": 95.29, "GAT": 95.07, "TriHetGCN": 97.24
            }
        }
    }
    
    # Prepare comparison data
    comparison_data = []
    paper_data = paper_results.get(dataset_name, {})
    
    for model, (auc, ap) in results.items():
        paper_auc = paper_data.get("AUC", {}).get(model, None)
        paper_ap = paper_data.get("AP", {}).get(model, None)
        
        if paper_auc is not None and paper_ap is not None:
            comparison_data.append({
                "Model": model,
                "Our AUC": auc * 100,
                "Paper AUC": paper_auc,
                "AUC Difference": (auc * 100) - paper_auc,
                "Our AP": ap * 100,
                "Paper AP": paper_ap,
                "AP Difference": (ap * 100) - paper_ap
            })
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    csv_path = f"{results_dir}/{dataset_name}_paper_comparison.csv"
    if not df.empty:
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved paper comparison table to {csv_path}")
    else:
        print(f"⚠️ No paper comparison data for {dataset_name}")
        return csv_path, None
    
    # Create visualization with new theme
    plt.figure(figsize=(14, 10))
    
    # Set theme colors
    our_color = '#66B3FF'  # Light blue
    paper_color = '#FFA500'  # Orange
    diff_color_positive = '#4CBB17'  # Green
    diff_color_negative = '#FF0000'  # Red
    bg_color = '#F5F5F5'
    text_color = '#333333'
    
    # Apply theme
    plt.style.use('ggplot')
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['axes.edgecolor'] = text_color
    plt.rcParams['axes.labelcolor'] = text_color
    plt.rcParams['text.color'] = text_color
    plt.rcParams['xtick.color'] = text_color
    plt.rcParams['ytick.color'] = text_color
    
    # Create positions for bars
    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    
    # AUC comparison
    plt.subplot(2, 1, 1)
    plt.bar(x - width/2, df["Our AUC"], width, label='Our Implementation', 
            color=our_color, edgecolor='darkblue', linewidth=1.2)
    plt.bar(x + width/2, df["Paper AUC"], width, label='Paper Results', 
            color=paper_color, edgecolor='darkorange', linewidth=1.2)
    
    # Add labels and title
    plt.title(f'AUC Comparison with Paper - {dataset_name}', 
              fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.ylabel('AUC (%)', fontsize=12, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend(frameon=True, framealpha=0.9, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, row in df.iterrows():
        plt.text(x[i] - width/2, row["Our AUC"] + 1, f'{row["Our AUC"]:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(x[i] + width/2, row["Paper AUC"] + 1, f'{row["Paper AUC"]:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        diff = row["AUC Difference"]
        diff_color = diff_color_positive if diff >= 0 else diff_color_negative
        diff_text = f'{"+" if diff >= 0 else ""}{diff:.2f}%'
        
        y_position = min(row["Our AUC"], row["Paper AUC"]) - 5
        if y_position < 0:
            y_position = 5
            
        plt.text(x[i], y_position, diff_text, 
                 ha='center', va='top', fontsize=10, 
                 fontweight='bold', color=diff_color)
    
    # AP comparison
    plt.subplot(2, 1, 2)
    plt.bar(x - width/2, df["Our AP"], width, label='Our Implementation', 
            color=our_color, edgecolor='darkblue', linewidth=1.2)
    plt.bar(x + width/2, df["Paper AP"], width, label='Paper Results', 
            color=paper_color, edgecolor='darkorange', linewidth=1.2)
    
    # Add labels and title
    plt.title(f'AP Comparison with Paper - {dataset_name}', 
              fontsize=16, fontweight='bold', pad=20, color=text_color)
    plt.ylabel('AP (%)', fontsize=12, fontweight='bold')
    plt.xticks(x, models, rotation=45, ha='right')
    plt.legend(frameon=True, framealpha=0.9, loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add value labels
    for i, row in df.iterrows():
        plt.text(x[i] - width/2, row["Our AP"] + 1, f'{row["Our AP"]:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        plt.text(x[i] + width/2, row["Paper AP"] + 1, f'{row["Paper AP"]:.2f}%', 
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        diff = row["AP Difference"]
        diff_color = diff_color_positive if diff >= 0 else diff_color_negative
        diff_text = f'{"+" if diff >= 0 else ""}{diff:.2f}%'
        
        y_position = min(row["Our AP"], row["Paper AP"]) - 5
        if y_position < 0:
            y_position = 5
            
        plt.text(x[i], y_position, diff_text, 
                 ha='center', va='top', fontsize=10, 
                 fontweight='bold', color=diff_color)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save and close
    plot_path = f"{results_dir}/{dataset_name}_paper_comparison.png"
    plt.savefig(plot_path, dpi=300, facecolor=bg_color)
    plt.close()
    print(f"💾 Saved paper comparison visualization to {plot_path}")
    
    return csv_path, plot_path