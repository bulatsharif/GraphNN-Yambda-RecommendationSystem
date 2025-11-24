from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from torch_geometric.data import Data
from typing import Tuple, Dict, List
import pandas as pd
import numpy as np
import torch

def get_mapping(df: pd.DataFrame, col: str) -> Tuple[Dict, int]:
    unique_ids = df[col].unique()
    id_to_idx = {id: idx for idx, id in enumerate(unique_ids)}
    return id_to_idx, unique_ids.shape[0]

def create_pyg_data(train_df, val_df, test_df, event_types_list) -> Tuple[Data, Dict, Dict, int, int]:
    unique_users = pd.unique(pd.concat([train_df['uid'], val_df['uid'], test_df['uid']]))
    unique_items = pd.unique(pd.concat([train_df['item_id'], val_df['item_id'], test_df['item_id']]))
    
    user_map = {id: idx for idx, id in enumerate(unique_users)}
    item_map = {id: idx for idx, id in enumerate(unique_items)}
    num_users = len(unique_users)
    num_items = len(unique_items)
    event_map = {etype: idx for idx, etype in enumerate(event_types_list)}
    
    print(f"Total Nodes: {num_users + num_items} (Users: {num_users}, Items: {num_items})")

    src_u = train_df['uid'].map(user_map).values
    dst_i = train_df['item_id'].map(item_map).values
    edge_type_vals = train_df['event_type'].map(event_map).values.astype(int)
    item_offset = num_users
    u_i_edges = np.stack([src_u, dst_i + item_offset])
    user_to_item_edge = torch.from_numpy(u_i_edges).long()
    
    i_u_edges = np.stack([dst_i + item_offset, src_u])
    item_to_user_edge = torch.from_numpy(i_u_edges).long()
    
    edge_type = torch.tensor(edge_type_vals, dtype=torch.long)
    edge_index = torch.cat([user_to_item_edge, item_to_user_edge], dim=1)
    edge_type = torch.cat([edge_type, edge_type], dim=0)

    data = Data(edge_index=edge_index, edge_type=edge_type, num_nodes=num_users + num_items)
    data.user_map = user_map
    data.item_map = item_map
    data.num_users = num_users
    data.num_items = num_items
    data.item_offset = item_offset
    data.num_behaviours = len(event_types_list)

    return data, user_map, item_map, num_users, num_items

def create_loaders(data, train_df, user_map, item_map, batch_size=1024, num_neighbors=[10, 5], target_event=None):
    item_offset = data.num_users
    
    def df_to_edge_label_index(df):
        u = df['uid'].map(user_map).values
        i = df['item_id'].map(item_map).values + item_offset
        stacked = np.stack([u, i])
        return torch.from_numpy(stacked).long()

    print(f"Creating Train Loader for target: {target_event}")
    train_target_df = train_df[train_df['event_type'] == target_event]
    
    if train_target_df.empty:
        raise ValueError(f"No training data found for target event '{target_event}'")

    train_edge_label_index = df_to_edge_label_index(train_target_df)
    
    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=train_edge_label_index,
        neg_sampling_ratio=1.0,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        subgraph_type='bidirectional' 
    )

    inference_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=4096,
        input_nodes=None, 
        shuffle=False,
        num_workers=2
    )
    return {"train": train_loader, "inference": inference_loader}
