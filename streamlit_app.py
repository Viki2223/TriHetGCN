import streamlit as st
import pandas as pd
from PIL import Image
import os
import glob
import numpy as np
import json

st.set_page_config(page_title="TriHetGCN Dashboard", layout="wide")
st.title("📊 TriHetGCN Link Prediction Results")

# Dataset selector
dataset = st.selectbox("📂 Choose Dataset", ["Cora", "CiteSeer", "PubMed"])
base_dir = f"results/{dataset}/checkpoints"

# Display graph
st.header("📌 Graph Structure with Hub Nodes")
graph_img = f"{base_dir}/graph_{dataset}.png"
if os.path.exists(graph_img):
    st.image(Image.open(graph_img), 
             caption=f"{dataset} Graph Structure", 
             use_container_width=True)
else:
    st.warning(f"⚠️ Graph image not found at {graph_img}")

# Display model loss curves
st.header("📉 Training Loss Curves")
loss_files = glob.glob(f"{base_dir}/{dataset}_loss_*.png")
if loss_files:
    cols = st.columns(2)
    for i, img_path in enumerate(loss_files):
        model_name = os.path.basename(img_path).split('_')[-1].split('.')[0]
        with cols[i % 2]:
            st.image(Image.open(img_path), 
                     caption=f"{model_name} Training Loss",
                     use_container_width=True)
else:
    st.warning(f"⚠️ No loss curves found in {base_dir}")

# Display AUC/AP bar plots
st.header("📈 Performance Metrics Comparison")
col1, col2 = st.columns(2)
with col1:
    auc_img = f"{base_dir}/{dataset}_AUC.png"
    if os.path.exists(auc_img):
        st.image(Image.open(auc_img), 
                 caption="AUC Comparison",
                 use_container_width=True)
    else:
        st.warning(f"AUC plot missing at {auc_img}")
with col2:
    ap_img = f"{base_dir}/{dataset}_AP.png"
    if os.path.exists(ap_img):
        st.image(Image.open(ap_img), 
                 caption="AP Comparison",
                 use_container_width=True)
    else:
        st.warning(f"AP plot missing at {ap_img}")

# Top Performers Section
st.header("🏆 Top Performers")

# Load overall results
summary_path = "results/overall_summary.json"
if os.path.exists(summary_path):
    with open(summary_path) as f:
        summary = json.load(f)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Best AUC")
        st.metric(label="Highest AUC Score", value=f"↑ {summary['best_auc']:.2f}%")
        st.caption(f"Achieved by {summary['model_auc']} on {summary['dataset_auc']} dataset")
        
    with col2:
        st.subheader("Best AP")
        st.metric(label="Highest AP Score", value=f"↑ {summary['best_ap']:.2f}%")
        st.caption(f"Achieved by {summary['model_ap']} on {summary['dataset_ap']} dataset")
    
    st.subheader("TriHetGCN Performance")
    col3, col4 = st.columns(2)
    with col3:
        st.metric(label="Average AUC", value=f"↑ {summary['avg_auc']:.2f}%")
    with col4:
        st.metric(label="Average AP", value=f"↑ {summary['avg_ap']:.2f}%")
    
    progress_value = int((summary['avg_auc'] + summary['avg_ap']) / 2)
    st.progress(progress_value)
    st.caption("Overall performance across all datasets")
else:
    st.warning("Overall summary not found. Please run training first.")

# Paper Comparison Table
paper_csv = f"{base_dir}/{dataset}_paper_comparison.csv"
if os.path.exists(paper_csv):
    df_paper = pd.read_csv(paper_csv)
    
    def color_diff(val):
        if val >= 0:
            return 'color: green; font-weight: bold'
        else:
            return 'color: red; font-weight: bold'
    
    styled_df = df_paper.style.applymap(color_diff, 
                                      subset=['AUC Difference', 'AP Difference'])
    
    st.subheader("🔍 Detailed Comparison with Paper")
    st.dataframe(styled_df.format({
        "Our AUC": "{:.2f}%",
        "Paper AUC": "{:.2f}%",
        "AUC Difference": "{:+.2f}%",
        "Our AP": "{:.2f}%",
        "Paper AP": "{:.2f}%",
        "AP Difference": "{:+.2f}%"
    }))
else:
    st.warning(f"⚠️ Paper comparison table not found at {paper_csv}")

# Paper Comparison
st.header("📚 Comparison with Published Results")
paper_comparison_img = f"{base_dir}/{dataset}_paper_comparison.png"
if os.path.exists(paper_comparison_img):
    st.image(Image.open(paper_comparison_img), 
             caption=f"Paper Comparison - {dataset}",
             use_container_width=True)
else:
    st.warning(f"⚠️ Paper comparison visualization not found at {paper_comparison_img}")

# Footer
st.markdown("---")
st.caption("🔍 Built using PyTorch Geometric + Streamlit | © Your Research Project")
