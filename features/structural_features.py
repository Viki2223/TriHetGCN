import torch
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data
import networkx as nx
import random

def compute_anchor_features(edge_index, num_nodes, anchors=5):
    anchor_features = torch.zeros((num_nodes, anchors))

    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)

    # Ensure all nodes exist
    missing = set(range(num_nodes)) - set(G.nodes())
    for node in missing:
        G.add_node(node)

    # ✅ Choose only available nodes as anchors
    available_nodes = list(G.nodes())
    anchor_nodes = torch.tensor(random.sample(available_nodes, anchors))

    for i, anchor in enumerate(anchor_nodes):
        lengths = nx.single_source_shortest_path_length(G, anchor.item())
        for node in range(num_nodes):
            dist = lengths.get(node, float('inf'))
            anchor_features[node, i] = 1.0 / (dist + 1)
    return anchor_features

def compute_cn_hi(edge_index, num_nodes):
    data = Data(edge_index=edge_index, num_nodes=num_nodes)
    G = to_networkx(data, to_undirected=True)

    for node in range(num_nodes):
        if node not in G:
            G.add_node(node)

    cn_dict = {}
    hi_dict = {}

    for u, v in G.edges():
        u_deg = G.degree[u]
        v_deg = G.degree[v]
        cn = len(set(G.neighbors(u)) & set(G.neighbors(v)))
        hi = abs(u_deg - v_deg)
        cn_dict[(u, v)] = torch.tensor(cn, dtype=torch.float32)
        hi_dict[(u, v)] = torch.tensor(hi, dtype=torch.float32)
    return cn_dict, hi_dict
