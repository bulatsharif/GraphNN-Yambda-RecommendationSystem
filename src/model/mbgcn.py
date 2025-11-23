from torch_geometric.nn.conv import SAGEConv
from typing import Literal, Optional
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn
import torch

class BehaviourGNNBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_layers: int = 1, dropout: int = 0.2):
        super().__init__()
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            input_dim = in_dim if i == 0 else hidden_dim
            self.convs.append(SAGEConv(input_dim, hidden_dim))

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers)])
        self.dropout_proba = dropout

    def forward(self, x: Tensor, edge_index: Tensor):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            if i < self.n_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout_proba, training=self.training)
        return x
    
class MBGCN(nn.Module):
    def __init__(
            self,
            num_users: int,
            num_items: int,
            num_behaviours: int,
            item_feat_dim: int = 128,
            node_emb_dim: int = 128,
            user_emb_dim: int = 128,
            behaviour_hidden_dim: int = 128,
            n_gnn_layers: int = 2,
            fusion: Literal['attention', 'concat', 'sum'] = 'attention',
            dropout: int = 0.2,
            use_pretrained_item_feats: bool = True
    ):
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.num_behaviours = num_behaviours
        self.node_emb_dim = node_emb_dim
        self.use_pretrained_item_feats = use_pretrained_item_feats

        self.user_emb = nn.Embedding(num_users, user_emb_dim)

        if not use_pretrained_item_feats:
            self.item_emb = nn.Embedding(num_items, item_feat_dim)
        else:
            self.item_emb = None

        self.user_proj = nn.Linear(user_emb_dim, node_emb_dim)
        self.item_proj = nn.Linear(item_feat_dim, node_emb_dim)
        self.fusion = fusion

        self.behaviour_blocks = nn.ModuleList([
            BehaviourGNNBlock(node_emb_dim, behaviour_hidden_dim, n_layers=n_gnn_layers, dropout=dropout)
            for _ in range(num_behaviours)
        ])

        if behaviour_hidden_dim != node_emb_dim:
            self.behav_proj = nn.ModuleList([
                nn.Linear(behaviour_hidden_dim, node_emb_dim) for _ in range(num_behaviours)
            ])
        else:
            self.behav_proj = None
        
        allowed_fusions = ['attention', 'concat', 'sum']
        assert fusion in allowed_fusions, f'fusion must be one of {allowed_fusions}'
        if fusion == "attention":
            self.query_proj = nn.Linear(node_emb_dim, node_emb_dim)
            self.key_projs = nn.ModuleList([nn.Linear(node_emb_dim, node_emb_dim) for _ in range(self.num_behaviours)])
            self.fuse_linear = nn.Linear(node_emb_dim, node_emb_dim)
        elif fusion == "concat":
            self.fuse_linear = nn.Linear(self.num_behaviours * node_emb_dim, node_emb_dim)
        else:
            self.fuse_linear = nn.Linear(node_emb_dim, node_emb_dim)

        self.refine = nn.Sequential(
            nn.Linear(node_emb_dim, node_emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.user_emb.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                 nn.init.xavier_uniform_(m.weight)

    def _get_node_features(self, node_ids: Optional[Tensor], external_item_feats: Optional[Tensor] = None) -> Tensor:
        """
        Builds initial nodes' features.
        If node_idx is None, then returns the features of all nodes in the graph
        """
        assert self.use_pretrained_item_feats == False or external_item_feats is not None
        device = self.user_emb.weight.device

        if node_ids is not None:
            is_user = node_ids < self.num_users
            user_ids = node_ids[is_user]
            item_ids = node_ids[~is_user] - self.num_users

            user_feats = self.user_emb(user_ids)
            if self.use_pretrained_item_feats:
                item_feats = external_item_feats[item_ids]
            else:
                item_feats = self.item_emb(item_ids)
            user_node_emb = self.user_proj(user_feats)
            item_node_emb = self.item_proj(item_feats)
            
            x = torch.empty((len(node_ids), self.node_emb_dim), device=device)
            x[is_user] = user_node_emb
            x[~is_user] = item_node_emb
            return x
        
        # return all node embeddings
        user_ids = torch.arange(self.num_users, device=device)
        item_ids = torch.arange(self.num_items, device=device)
        
        user_feats = self.user_emb(user_ids)
        if self.use_pretrained_item_feats:
            item_feats = external_item_feats
        else:
            item_feats = self.item_emb(item_ids)

        user_node_emb = self.user_proj(user_feats)
        item_node_emb = self.item_proj(item_feats)
        return torch.cat([user_node_emb, item_node_emb], dim=0)
    
    def forward(self, edge_index: Tensor, edge_type: Tensor, node_ids: Optional[Tensor] = None, item_feats: Optional[Tensor] = None):
        x = self._get_node_features(node_ids, item_feats)

        behaviour_outputs = []
        for behav_idx in range(self.num_behaviours):
            mask = (edge_type == behav_idx)
            if mask.sum() == 0:
                behaviour_outputs.append(x)
                continue
        
            edge_index_b = edge_index[:, mask]
            h = self.behaviour_blocks[behav_idx](x, edge_index_b)

            if self.behav_proj is not None:
                h = self.behav_proj[behav_idx](h)
            
            h = x + h
            behaviour_outputs.append(h)
        
        stack = torch.stack(behaviour_outputs, dim=0)

        # Fusion logic
        if self.fusion_type == 'sum':
            fused = torch.sum(stack, dim=0)
            fused = self.fuse_linear(fused)
        elif self.fusion_type == 'concat':
            concat = stack.permute(1, 0, 2).contiguous().view(x.size(0), -1)
            fused = self.fuse_linear(concat)
        elif self.fusion_type == 'attention':
            Q = self.query_proj(x)
            attn_logits = []
            for b in range(self.num_behaviours):
                Kb = self.key_projs[b](stack[b])
                log = (Q * Kb).sum(dim=-1, keepdim=True)
                attn_logits.append(log)

            attn_logits = torch.cat(attn_logits, dim=-1)
            attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1)
            stack_perm = stack.permute(1, 0, 2)
            fused = (attn_weights * stack_perm).sum(dim=1)
            fused = self.fuse_linear(fused)
        else:
            raise ValueError(f"Unknown fusion: {self.fusion_type}")

        final_emb = self.refine(fused)
        return final_emb
