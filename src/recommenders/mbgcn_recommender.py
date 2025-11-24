import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.model.mbgcn import MBGCN
from src.datasets.dataloader import create_pyg_data
from src.recommenders.BaseRecommender import BaseRecommender

class MBGCNRecommender(BaseRecommender):
    def __init__(self, 
                 model_params: dict, 
                 item_features_path: str = None, 
                 checkpoint_path: str = None,
                 device: str = "cuda"):
        """
        Args:
            model_params: Dictionary of parameters for MBGCN __init__ (e.g., n_gnn_layers).
            item_features_path: Path to parquet file containing item embeddings/features.
            checkpoint_path: Path to a saved state_dict to load (optional).
            device: 'cuda' or 'cpu'.
        """
        self.model_params = model_params
        self.item_features_path = item_features_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        
        self.model = None
        self.user_map = None
        self.item_map = None
        self.idx_to_item_id = None
        
        # Cached Embeddings
        self.user_emb_matrix = None
        self.item_emb_matrix = None

    def fit(self, train_df: pd.DataFrame, test_df: pd.DataFrame = None, **kwargs):
        """
        Prepares the model for recommendation.
        1. Builds the graph (PyG Data) using train and test data.
        2. Loads features from parquet and aligns them with graph indices.
        3. Initializes the model and loads weights.
        4. Runs a forward pass to cache user/item embeddings.
        """
        # 1. Create Graph & Mappings
        # We pass test_df as val_df just to satisfy the signature and ensure all nodes are mapped
        val_df = test_df if test_df is not None else train_df
        test_df_arg = test_df if test_df is not None else train_df
        
        data, self.user_map, self.item_map, num_users, num_items = create_pyg_data(
            train_df, val_df, test_df_arg
        )
        
        # Create reverse mapping for outputting recommendations
        self.idx_to_item_id = {v: k for k, v in self.item_map.items()}

        # 2. Load and Align Item Features
        feat_dim = 128  # As specified in your schema
        
        # Initialize with Xavier uniform (random) for all items first.
        # This ensures that if some items in the graph don't have embeddings in the parquet,
        # they still have a valid starting point.
        item_feats = torch.empty((num_items, feat_dim))
        nn.init.xavier_uniform_(item_feats)

        if self.item_features_path:
            print(f"Loading embeddings from {self.item_features_path}")
            emb_df = pd.read_parquet(self.item_features_path)
            
            if 'item_id' not in emb_df.columns or 'normalized_embed' not in emb_df.columns:
                 raise ValueError("Embedding file must contain 'item_id' and 'normalized_embed' columns")
            
            # Map the dataframe's 'item_id' to the graph's internal 'index'
            # We use the item_map created during graph construction
            emb_df['graph_idx'] = emb_df['item_id'].map(self.item_map)
            
            # Filter out items that are in the parquet but NOT in our current graph
            valid_emb_df = emb_df.dropna(subset=['graph_idx'])
            
            if not valid_emb_df.empty:
                # Get the indices (integers)
                indices = valid_emb_df['graph_idx'].astype(int).values
                indices_tensor = torch.LongTensor(indices)
                
                # Stack the arrays from the 'normalized_embed' column
                # This converts the Series of arrays into a single 2D numpy array
                features_array = np.stack(valid_emb_df['normalized_embed'].values)
                features_tensor = torch.tensor(features_array, dtype=torch.float)
                
                # Assign the pre-trained embeddings to the correct rows in item_feats
                item_feats[indices_tensor] = features_tensor
                
                print(f"Successfully aligned and loaded embeddings for {len(indices)} items.")
            else:
                print("Warning: No overlap found between embedding file items and graph items.")
        
        # 3. Initialize Model
        self.model = MBGCN(
            num_users=num_users,
            num_items=num_items,
            num_behaviours=3, # Should be consistent with your data
            item_feat_dim=feat_dim,
            **self.model_params
        )
        
        if self.checkpoint_path:
            print(f"Loading checkpoint from {self.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.eval()

        # 4. Cache Embeddings (Forward Pass)
        print("Running forward pass to cache embeddings...")
        with torch.no_grad():
            final_emb = self.model(
                item_feats.to(self.device),
                data.edge_index.to(self.device),
                data.edge_type.to(self.device)
            )
            self.user_emb_matrix = final_emb[:num_users]
            self.item_emb_matrix = final_emb[num_users:]

    def recommend(self, user_ids: List[int], k: int, history_filter: Dict[int, List[int]] = None) -> Dict[int, List[int]]:
        """
        Generates recommendations by performing dot-product search on cached embeddings.
        """
        if self.user_emb_matrix is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
            
        recommendations = {}
        batch_size = 1024
        
        valid_users = [u for u in user_ids if u in self.user_map]
        
        for i in range(0, len(valid_users), batch_size):
            batch_uids_original = valid_users[i : i + batch_size]
            
            # Map to indices
            batch_uids_idx = [self.user_map[u] for u in batch_uids_original]
            batch_uids_tensor = torch.tensor(batch_uids_idx, device=self.device, dtype=torch.long)
            
            # Get Embeddings & Score
            batch_user_emb = self.user_emb_matrix[batch_uids_tensor] 
            scores = torch.matmul(batch_user_emb, self.item_emb_matrix.t()) 
            
            # Mask History
            if history_filter:
                for idx, uid_orig in enumerate(batch_uids_original):
                    if uid_orig in history_filter:
                        hist_items = [self.item_map[it] for it in history_filter[uid_orig] if it in self.item_map]
                        if hist_items:
                            # Using -inf ensures these items are never selected in topk
                            scores[idx, hist_items] = -float('inf')

            # Top K
            _, top_indices = torch.topk(scores, k=k, dim=1)
            top_indices = top_indices.cpu().numpy()
            
            # Map back to Original IDs
            for idx, uid_orig in enumerate(batch_uids_original):
                recommendations[uid_orig] = [self.idx_to_item_id[idx] for idx in top_indices[idx]]
                
        return recommendations