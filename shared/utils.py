import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import json
from collections import defaultdict

def save_model(model, path):
    torch.save(model.state_dict(), path)

def plot_metrics(results, dataset_name, results_dir):
    models = list(results.keys())
    aucs = [results[m][0] * 100 for m in models]  # Convert to percentage
    aps = [results[m][1] * 100 for m in models]    # Convert to percentage
    
    # AUC Plot
    plt.figure(figsize=(6,4))
    plt.bar(models, aucs, color="skyblue")
    plt.ylabel("AUC Score (%)")
    plt.title(f"{dataset_name} AUC Scores")
    plt.ylim(0, 100)  # Set y-axis to 0-100%
    auc_path = f"{results_dir}/{dataset_name}_AUC.png"
    plt.savefig(auc_path)
    plt.close()
    
    # AP Plot
    plt.figure(figsize=(6,4))
    plt.bar(models, aps, color="lightgreen")
    plt.ylabel("Average Precision (%)")
    plt.title(f"{dataset_name} AP Scores")
    plt.ylim(0, 100)  # Set y-axis to 0-100%
    ap_path = f"{results_dir}/{dataset_name}_AP.png"
    plt.savefig(ap_path)
    plt.close()
    print(f"💾 Saved metrics plots to {auc_path} and {ap_path}")

def save_results_table(results, dataset_name, results_dir):
    # Convert results to DataFrame
    df = pd.DataFrame(results).T
    df.columns = ["AUC", "AP"]
    
    # Format display values
    df_display = df.copy()
    df_display["AUC"] = df_display["AUC"].apply(lambda x: f"{x*100:.2f}%")
    df_display["AP"] = df_display["AP"].apply(lambda x: f"{x*100:.2f}%")
    
    # Save paths
    raw_path = f"{results_dir}/{dataset_name}_results_table.csv"
    display_path = f"{results_dir}/{dataset_name}_results_table_display.csv"
    
    df.to_csv(raw_path)
    df_display.to_csv(display_path)
    print(f"💾 Saved results tables to {raw_path} and {display_path}")
    
    # Return formatted table for printing
    return df_display

def save_overall_results():
    datasets = ["Cora", "CiteSeer", "PubMed"]
    overall_results = {"TriHetGCN": {"AUC": [], "AP": []}}
    best_results = {
        "AUC": 0, 
        "AP": 0, 
        "model_auc": "", 
        "model_ap": "", 
        "dataset_auc": "", 
        "dataset_ap": ""
    }
    
    for ds in datasets:
        path = f"results/{ds}/{ds}_results_table.csv"
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            
            # Collect TriHetGCN results
            if "TriHetGCN" in df.index:
                auc_val = df.loc["TriHetGCN", "AUC"]
                ap_val = df.loc["TriHetGCN", "AP"]
                overall_results["TriHetGCN"]["AUC"].append(auc_val)
                overall_results["TriHetGCN"]["AP"].append(ap_val)
            
            # Track best performers
            auc_max = df["AUC"].max()
            ap_max = df["AP"].max()
            if auc_max > best_results["AUC"]:
                best_results["AUC"] = auc_max
                best_results["model_auc"] = df["AUC"].idxmax()
                best_results["dataset_auc"] = ds
            if ap_max > best_results["AP"]:
                best_results["AP"] = ap_max
                best_results["model_ap"] = df["AP"].idxmax()
                best_results["dataset_ap"] = ds
    
    # Calculate averages for TriHetGCN
    avg_auc = np.mean(overall_results["TriHetGCN"]["AUC"]) * 100
    avg_ap = np.mean(overall_results["TriHetGCN"]["AP"]) * 100
    
    # Save to file
    summary = {
        "best_auc": best_results["AUC"] * 100,
        "best_ap": best_results["AP"] * 100,
        "model_auc": best_results["model_auc"],
        "model_ap": best_results["model_ap"],
        "dataset_auc": best_results["dataset_auc"],
        "dataset_ap": best_results["dataset_ap"],
        "avg_auc": avg_auc,
        "avg_ap": avg_ap
    }
    
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    with open("results/overall_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    
    print(f"💾 Saved overall summary to results/overall_summary.json")
    return summary

# Add performance comparison functions
from .performance_comparison import plot_performance_comparison
from .paper_comparison import compare_with_paper