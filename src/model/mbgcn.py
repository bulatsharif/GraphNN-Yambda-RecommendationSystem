from torch_geometric.nn.conv import SAGEConv
from typing import Literal
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
            dropout: int = 0.2):
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items
        self.num_behaviours = num_behaviours
        self.node_emb_dim = node_emb_dim

        self.user_emb = nn.Embedding(num_users, user_emb_dim)
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

    def _build_initial_node_features(self, item_feats: Tensor) -> Tensor:
        device = item_feats.device
        users = self.user_emb.weight
        users = self.user_proj(users)
        items = self.item_proj(item_feats)
        x = torch.cat([users, items], dim=0)
        return x.to(device)
    
    def forward(self, item_feats: Tensor, edge_index: Tensor, edge_type: Tensor):
        x = self._build_initial_node_features(item_feats)

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

        if self.fusion == 'sum':
            fused = torch.sum(stack, dim=0)
            fused = self.fuse_linear(fused)
        elif self.fusion == 'concat':
            concat = stack.permute(1, 0, 2).contiguous().view(self.num_nodes, -1)
            fused = self.fuse_linear(concat)
        elif self.fusion == 'attention':
            Q = self.query_proj(x)
            attn_logits = []
            for b in range(self.num_behaviours):
                Kb = self.key_projs[b](stack[b])
                log = (Q * Kb).sum(dim=-1, keepdim=True)
                attn_logits.append(log)

            attn_logits = torch.cat(attn_logits, dim=-1)
            attn_weights = F.softmax(attn_logits, dim=-1)

            attn_weights = attn_weights.unsqueeze(-1)
            stack_perm = stack.permute(1, 0, 2)
            fused = (attn_weights * stack_perm).sum(dim=1)
            fused = self.fuse_linear(fused)
        else:
            raise RuntimeError(f'Unknown fusion mechanism: {self.fusion}')

        final_emb = self.refine(fused)
        return final_emb
