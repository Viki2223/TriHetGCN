import matplotlib.pyplot as plt
import networkx as nx
import torch
import random

def visualize_graph_structure(edge_index, labels, dataset_name, save_path):
    # Convert to networkx graph
    G = nx.Graph()
    edge_list = edge_index.t().tolist()
    G.add_edges_from(edge_list)
    
    # Identify hub nodes (top 10% by degree)
    degrees = dict(G.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    if sorted_degrees:
        hub_threshold = sorted_degrees[len(G)//10][1]  # Top 10%
        hub_nodes = [n for n, d in degrees.items() if d >= hub_threshold]
    else:
        hub_nodes = []
    
    # Sample subgraph for large datasets
    if G.number_of_nodes() > 500:
        # Ensure hub nodes are included
        non_hubs = [n for n in G.nodes() if n not in hub_nodes]
        sampled_non_hubs = random.sample(non_hubs, min(400, len(non_hubs)))
        sampled_nodes = hub_nodes + sampled_non_hubs
        G = G.subgraph(sampled_nodes)
        print(f"📊 Visualizing sampled subgraph of {len(sampled_nodes)} nodes for {dataset_name}")
    
    # Use spring layout with hub nodes at center
    pos = nx.spring_layout(G, seed=42, k=0.15, iterations=50)
    
    plt.figure(figsize=(10, 8))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    # Draw non-hub nodes
    non_hub_nodes = [n for n in G.nodes() if n not in hub_nodes]
    if non_hub_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=non_hub_nodes,
            node_size=20,
            node_color="lightblue",
            edgecolors="gray",
            linewidths=0.5
        )
    
    # Draw hub nodes
    if hub_nodes:
        nx.draw_networkx_nodes(
            G, pos, nodelist=hub_nodes,
            node_size=200,
            node_color="coral",
            edgecolors="darkred",
            linewidths=1.0,
            alpha=0.8
        )
    
    plt.title(f"{dataset_name} Graph Structure (Hub Nodes in Coral)", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"💾 Saved graph visualization to {save_path}")