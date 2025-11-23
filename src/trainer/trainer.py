import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any, Optional
import numpy as np
import scipy.sparse as sp

from src.metrics.ranking_metrics import RankingEvaluator

class MBGCNTrainer:
    def __init__(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        criterion: nn.Module, 
        loaders: Dict,
        full_data: Any,
        evaluator: RankingEvaluator,
        val_ground_truth: sp.csr_matrix,
        logger, 
        device: str, 
        item_feats: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = loaders['train']
        self.full_data = full_data
        self.evaluator = evaluator
        self.val_ground_truth = val_ground_truth
        self.logger = logger
        self.device = device
        self.item_feats = item_feats
        
        self.step = 0

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Train]", leave=False)
        
        for batch in pbar:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(
                edge_index=batch.edge_index, 
                edge_type=batch.edge_type, 
                node_ids=batch.n_id,
                item_feats=self.item_feats
            )
            
            # Calculate loss
            pos_mask = batch.edge_label == 1.0
            neg_mask = batch.edge_label == 0.0
            pos_edge_index = batch.edge_label_index[:, pos_mask]
            neg_edge_index = batch.edge_label_index[:, neg_mask]
            assert pos_edge_index.shape == neg_edge_index.shape, "BPR loss expects negative_sampling_ratio=1.0!"

            pos_user_emb = out[pos_edge_index[0]]
            pos_item_emb = out[pos_edge_index[1]]
            pos_scores = (pos_user_emb * pos_item_emb).sum(dim=-1)

            neg_user_emb = out[neg_edge_index[0]]
            neg_item_emb = out[neg_edge_index[1]]
            neg_scores = (neg_user_emb * neg_item_emb).sum(dim=-1)
                            
            loss = self.criterion(pos_scores, neg_scores)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.step += 1
            
            if self.step % 50 == 0:
                self.logger.log_metrics({'Train/Loss': loss.item()}, self.step)
                
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, batch_size=512):
        self.model.eval()
        full_data_gpu = self.full_data.to(self.device)
        
        all_embs = self.model(
            edge_index=full_data_gpu.edge_index,
            edge_type=full_data_gpu.edge_type,
            node_ids=None, 
            item_feats=self.item_feats
        )
        
        num_users = self.model.num_users
        user_embs_all = all_embs[:num_users]
        item_embs_all = all_embs[num_users:]
        
        val_users = np.unique(self.val_ground_truth.nonzero()[0])
        n_val = len(val_users)
        
        metrics_accum = {f'ndcg@{k}': 0.0 for k in self.evaluator.k_list}
        if 'recall' in self.evaluator.metric_names:
             for k in self.evaluator.k_list: metrics_accum[f'recall@{k}'] = 0.0
        
        num_batches = (n_val + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_val)
            batch_users = val_users[start:end]
            
            u_emb = user_embs_all[batch_users] 
            batch_gt_sparse = self.val_ground_truth[batch_users]
            batch_gt_dense = torch.FloatTensor(batch_gt_sparse.toarray()).to(self.device)
            
            results = self.evaluator.evaluate_full_ranking(
                user_emb=u_emb,
                ground_truth_mask=batch_gt_dense,
                all_item_emb=item_embs_all
            )
            
            for k, v in results.items():
                metrics_accum[k] += v * len(batch_users)
                
        final_metrics = {k: v / n_val for k, v in metrics_accum.items()}
        prefixed = {f'Val/{k}': v for k, v in final_metrics.items()}
        self.logger.log_metrics(prefixed, self.step)
        
        return final_metrics
