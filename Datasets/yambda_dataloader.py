import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

def get_mapping(df, col):
    """Maps unique IDs in a column to a contiguous range [0, N-1]."""
    unique_ids = df[col].unique()
    id_to_idx = {id: idx for idx, id in enumerate(unique_ids)}
    return id_to_idx, unique_ids.shape(0)

def create_pyg_data(train_df, val_df, test_df):
    """
    Converts train/val/test DataFrames into a single PyG Data object (Bipartite Graph).
    Training interactions form the graph structure. Test interactions are stored for prediction targets.
    """
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # 1. Global Mapping (ensures consistent IDs across all splits)
    user_map, num_users = get_mapping(all_df, 'uid')
    item_map, num_items = get_mapping(all_df, 'item_id')
    
    print(f"Total Nodes: {num_users + num_items} (Users: {num_users}, Items: {num_items})")

    # 2. Prepare Edges (using only train_df for the graph structure to avoid look-ahead leakage)
    src_u = train_df['uid'].map(user_map).values
    dst_i = train_df['item_id'].map(item_map).values
    
    # Item node indices are offset to be distinct from user node indices
    item_offset = num_users

    # Create Bi-directional edges: (User -> Item) and (Item -> User)
    user_to_item_edge = torch.tensor([src_u, dst_i + item_offset], dtype=torch.long)
    item_to_user_edge = torch.tensor([dst_i + item_offset, src_u], dtype=torch.long)
    
    edge_index = torch.cat([user_to_item_edge, item_to_user_edge], dim=1)
    
    # 3. Create PyG Data object
    test_uids_mapped = torch.tensor(test_df['uid'].map(user_map).values, dtype=torch.long)
    test_items_mapped = torch.tensor(test_df['item_id'].map(item_map).values, dtype=torch.long) + item_offset

    data = Data(
        edge_index=edge_index, 
        num_nodes=num_users + num_items,
        
        # Store test interactions as edges for evaluation (Link Prediction target)
        test_edge_index=torch.stack([test_uids_mapped, test_items_mapped], dim=0)
    )
    
    return data, num_users, num_items


def create_neighbor_loaders(data, num_users, batch_size=32, num_hops=2):
    """
    Creates the PyG NeighborLoader for efficient k-hop neighborhood sampling.
    """
    # num_neighbors defines the number of neighbors to sample for each hop/GNN layer.
    # [-1] means sampling ALL neighbors at that level (best for smaller, dense graphs, or first layer).
    num_neighbors = [-1] * num_hops 
    print(f"Using {num_hops} hops with {num_neighbors} neighbors sampled per hop.")

    # Use unique user IDs from the test set as the starting points (input_nodes) for the sampler.
    # We sample the subgraph centered around the users involved in the test interactions.
    test_uids_nodes = data.test_edge_index[0].unique()
    
    test_loader = NeighborLoader(
        data, 
        input_nodes=test_uids_nodes, 
        num_neighbors=num_neighbors, 
        batch_size=batch_size, 
        shuffle=False,
    )
    
    return test_loader