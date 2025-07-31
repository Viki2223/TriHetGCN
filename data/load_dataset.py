import torch

def load_custom_dataset(dataset_name):
    if dataset_name == "Cora":
        edge_path = "Datasets/Cora/cora.edges"
        label_path = "Datasets/Cora/cora.node_labels"
    elif dataset_name == "CiteSeer":
        edge_path = "Datasets/CiteSeer/citeseer.edges"
        label_path = "Datasets/CiteSeer/citeseer.node_labels"
    elif dataset_name == "PubMed":
        edge_path = "Datasets/PubMed/PubMed.edges"
        label_path = "Datasets/PubMed/PubMed.node_labels"
    else:
        raise ValueError("Unknown dataset")

    # Read and parse edges
    edge_list = []
    max_node_id = 0
    with open(edge_path, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                src, tgt = int(parts[0]), int(parts[1])
                edge_list.append([src, tgt])
                max_node_id = max(max_node_id, src, tgt)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    # Read labels robustly
    labels = []
    with open(label_path, 'r') as f:
        for idx, line in enumerate(f):
            val = line.strip().split(",")[0]  # allow CSV or single label
            if val.isdigit():
                labels.append(int(val))
            else:
                print(f"⚠️  Skipping non-integer label at line {idx + 1}: {line.strip()}")

    if len(labels) == 0:
        print(f"❌ No valid labels found in {label_path}. Please check file formatting.")
        return edge_index, torch.tensor([], dtype=torch.long), max_node_id + 1

    labels = torch.tensor(labels, dtype=torch.long)
    num_nodes = max(max_node_id + 1, len(labels))
    print(f"✅ Loaded {len(labels)} labels and {edge_index.shape[1]} edges for {dataset_name}")

    return edge_index, labels, num_nodes