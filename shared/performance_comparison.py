import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib as mpl

def plot_performance_comparison(results, dataset_name, results_dir):
    # Extract model names and metrics
    models = list(results.keys())
    aucs = [results[m][0] * 100 for m in models]  # Convert to percentage
    aps = [results[m][1] * 100 for m in models]   # Convert to percentage
    
    # Find top performers
    top_auc_idx = np.argmax(aucs)
    top_ap_idx = np.argmax(aps)
    top_auc_model = models[top_auc_idx]
    top_ap_model = models[top_ap_idx]
    
    # Find TriHetGCN performance
    trihet_auc = next((auc for model, auc in zip(models, aucs) if "TriHetGCN" in model), 0)
    trihet_ap = next((ap for model, ap in zip(models, aps) if "TriHetGCN" in model), 0)
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Custom styling
    plt.style.use('seaborn-whitegrid')
    mpl.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333F4B'
    plt.rcParams['axes.linewidth'] = 0.8
    
    # Create positions for bars
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    auc_bars = plt.bar(x - width/2, aucs, width, label='AUC', 
                       color='#4C72B0', edgecolor='darkblue', linewidth=1.2)
    ap_bars = plt.bar(x + width/2, aps, width, label='AP', 
                      color='#55A868', edgecolor='darkgreen', linewidth=1.2)
    
    # Highlight top performers
    auc_bars[top_auc_idx].set_color('#FF7F0E')
    auc_bars[top_auc_idx].set_edgecolor('darkorange')
    ap_bars[top_ap_idx].set_color('#FF7F0E')
    ap_bars[top_ap_idx].set_edgecolor('darkorange')
    
    # Add labels and title
    plt.xlabel('Models', fontsize=12, fontweight='bold', color='#333F4B')
    plt.ylabel('Score (%)', fontsize=12, fontweight='bold', color='#333F4B')
    plt.title(f'Model Performance Comparison - {dataset_name}', 
              fontsize=16, fontweight='bold', pad=20, color='#333F4B')
    
    # Set x-axis ticks
    plt.xticks(x, models, rotation=45, ha='right')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(frameon=True, framealpha=0.9, loc='upper right')
    
    # Add annotations for top performers
    plt.annotate(f'Best AUC\n↑ {aucs[top_auc_idx]:.2f}%', 
                 xy=(x[top_auc_idx] - width/2, aucs[top_auc_idx] + 1),
                 xytext=(0, 15), textcoords='offset points',
                 ha='center', va='bottom', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#FF7F0E'))
    
    plt.annotate(f'Best AP\n↑ {aps[top_ap_idx]:.2f}%', 
                 xy=(x[top_ap_idx] + width/2, aps[top_ap_idx] + 1),
                 xytext=(0, 15), textcoords='offset points',
                 ha='center', va='bottom', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='#FF7F0E'))
    
    # Add TriHetGCN annotations
    if trihet_auc > 0:
        trihet_idx = next(i for i, m in enumerate(models) if "TriHetGCN" in m)
        plt.annotate(f'TriHetGCN AUC\n↑ {trihet_auc:.2f}%', 
                     xy=(x[trihet_idx] - width/2, trihet_auc + 1),
                     xytext=(0, -30), textcoords='offset points',
                     ha='center', va='top', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='#4C72B0'))
        
        plt.annotate(f'TriHetGCN AP\n↑ {trihet_ap:.2f}%', 
                     xy=(x[trihet_idx] + width/2, trihet_ap + 1),
                     xytext=(0, -30), textcoords='offset points',
                     ha='center', va='top', fontweight='bold',
                     arrowprops=dict(arrowstyle='->', lw=1.5, color='#55A868'))
    
    # Add value labels
    for i, v in enumerate(aucs):
        plt.text(x[i] - width/2, v + 1, f'{v:.2f}%', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    for i, v in enumerate(aps):
        plt.text(x[i] + width/2, v + 1, f'{v:.2f}%', 
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Adjust layout
    plt.ylim(0, max(max(aucs), max(aps)) * 1.25)
    plt.tight_layout()
    
    # Save and close
    save_path = f"{results_dir}/{dataset_name}_model_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"💾 Saved model comparison chart to {save_path}")
    
    return save_path