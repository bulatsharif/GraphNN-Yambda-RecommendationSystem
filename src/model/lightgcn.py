from torch_geometric.nn.conv import LGConv
from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor

class LightGCN(nn.Module):
    def __init__(
            self,
            num_users: int,
            num_items: int,
            embedding_dim: int = 128,
            n_layers: int = 3,
            use_pretrained_item_feats: bool = True,
            item_feat_dim: int = 128
    ):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.use_pretrained_item_feats = use_pretrained_item_feats

        # user embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)

        # item embeddings
        if not use_pretrained_item_feats:
            self.item_emb = nn.Embedding(num_items, embedding_dim)
            self.item_proj = None
        else:
            self.item_emb = None
            self.item_proj = nn.Linear(item_feat_dim, embedding_dim)

        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        if self.item_emb is not None:
            nn.init.xavier_uniform_(self.item_emb.weight)
        if self.item_proj is not None:
            nn.init.xavier_uniform_(self.item_proj.weight)
            if self.item_proj.bias is not None:
                nn.init.zeros_(self.item_proj.bias)

    def _get_node_features(self, node_ids: Tensor, item_feats: Optional[Tensor] = None) -> Tensor:
        """
        Retrieves embeddings for a batch of nodes (users + items).
        node_ids: Global node indices (0 to num_users+num_items-1)
        """
        device = node_ids.device
        
        is_user = node_ids < self.num_users
        user_ids = node_ids[is_user]
        item_ids = node_ids[~is_user] - self.num_users

        user_e = self.user_emb(user_ids)
        
        if self.use_pretrained_item_feats:
            if item_feats is None:
                 raise ValueError("Model requires item_feats but none provided.")
            batch_item_feats = item_feats[item_ids]
            item_e = self.item_proj(batch_item_feats)
        else:
            item_e = self.item_emb(item_ids)

        # Construct x for the batch
        x = torch.empty((len(node_ids), self.embedding_dim), device=device)
        x[is_user] = user_e
        x[~is_user] = item_e
        
        return x

    def forward(self, edge_index: Tensor, node_ids: Tensor, item_feats: Optional[Tensor] = None):
        """
        Forward pass for LightGCN on a subgraph (mini-batch) or full graph.
        
        Args:
            edge_index: Edges in the batch (remapped to 0..batch_size-1).
            node_ids: Global IDs of the nodes in this batch.
            item_feats: Pretrained item features (optional).
        """
        x = self._get_node_features(node_ids, item_feats)
        
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs.append(x)
        
        # LightGCN aggregation: mean of all layers
        x_final = torch.stack(xs, dim=0).mean(dim=0)
        return x_final
