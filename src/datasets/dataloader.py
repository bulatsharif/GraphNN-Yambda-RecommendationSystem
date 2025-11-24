from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.data import Data
from typing import Tuple, Dict, List
import pandas as pd
import torch

def get_mapping(df: pd.DataFrame, col: str) -> Tuple[Dict, int]:
    """Maps unique IDs in a column to a contiguous range [0, N-1]."""
    unique_ids = df[col].unique()
    id_to_idx = {id: idx for idx, id in enumerate(unique_ids)}
    return id_to_idx, unique_ids.shape(0)

def create_pyg_data(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame
) -> Tuple[Data, Dict, Dict, int, int]:
    """
    Converts train/val/test DataFrames into a single PyG Data object (Bipartite Graph).

    Returns:
        data: The graph object containing only training edges for message passing.
        user_map: Mapping from original user ID to index.
        item_map: Mapping from original item ID to index.
        num_users: Count of users.
        num_items: Count of items.
    """
    # 1. Global Mapping (ensures consistent IDs across all splits)
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
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

    # 3. Handle Edge Types
    behav_types = torch.tensor(train_df['event_type'].values, dtype=torch.long)
    edge_type = torch.cat([behav_types, behav_types], dim=0)

    data = Data(
        edge_index=edge_index,
        edge_type=edge_type,
        num_nodes=num_users + num_items
    )
    
    data.user_map = user_map
    data.item_map = item_map
    data.num_users = num_users
    data.num_items = num_items
    data.item_offset = item_offset

    return data, user_map, item_map, num_users, num_items

def create_loaders(
    data: Data,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    user_map: Dict,
    item_map: Dict,
    batch_size: int = 1024,
    num_neighbors: List[int] = [10, 5]
) -> Dict[str, LinkNeighborLoader]:
    """
    Creates LinkNeighborLoaders for train, val, and test splits.
    """
    item_offset = data.num_users
    
    def df_to_edge_label_index(df):
        u = torch.tensor(df['uid'].map(user_map).values, dtype=torch.long)
        i = torch.tensor(df['item_id'].map(item_map).values, dtype=torch.long) + item_offset
        return torch.stack([u, i], dim=0)

    # Train Loader
    train_edge_label_index = df_to_edge_label_index(train_df)
    
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=train_edge_label_index,
        neg_sampling_ratio=1.0,
        batch_size=batch_size,
        shuffle=True
    )

    # Validation Loader
    val_edge_label_index = df_to_edge_label_index(val_df)
    
    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=val_edge_label_index,
        batch_size=batch_size,
        shuffle=False 
    )

    # Inference Loader
    inference_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=4096,
        input_nodes=None,
        shuffle=False
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "inference": inference_loader
    }
