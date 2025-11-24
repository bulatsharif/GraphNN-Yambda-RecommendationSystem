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
        train_ground_truth: sp.csr_matrix,
        val_ground_truth: sp.csr_matrix,
        logger, 
        device: str, 
        item_feats: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.train_loader = loaders['train']
        self.val_loader = loaders['val'] 
        self.inference_loader = loaders['inference']

        self.evaluator = evaluator
        self.train_ground_truth = train_ground_truth
        self.val_ground_truth = val_ground_truth
        self.logger = logger
        self.device = device
        self.item_feats = item_feats.to(device) if item_feats is not None else None
        
        self.step = 0

    def _run_batch(self, batch):
        """Train step logic"""
        batch = batch.to(self.device)
        
        out = self.model(
            edge_index=batch.edge_index, 
            edge_type=batch.edge_type, 
            node_ids=batch.n_id,
            item_feats=self.item_feats
        )
        
        assert hasattr(batch, 'edge_label'), "Data batch must have 'edge_label' attribute to calculate loss"

        pos_mask = batch.edge_label == 1.0
        neg_mask = batch.edge_label == 0.0
        user_emb = out[batch.edge_label_index[0]]
        item_emb = out[batch.edge_label_index[1]]
        pos_scores = (user_emb[pos_mask] * item_emb[pos_mask]).sum(dim=-1)
        neg_scores = (user_emb[neg_mask] * item_emb[neg_mask]).sum(dim=-1)
        assert len(pos_scores) == len(neg_scores), "BPR loss assumes that negative_sampling_ratio = 1.0!"
        
        loss = self.criterion(pos_scores, neg_scores)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx} [Train]", leave=False)
        
        for batch in pbar:
            loss_val = self._run_batch(batch)
            total_loss += loss_val
            self.step += 1
            
            if self.step % 100 == 0:
                self.logger.log_metrics({'Train/Loss': loss_val}, self.step)
            pbar.set_postfix({'loss': loss_val})
            
        return total_loss / len(self.train_loader)
    
    @torch.no_grad()
    def get_all_embeddings(self) -> torch.Tensor:
        """
        Obtain the embeddings of all users and track on cpu through model forwarding by batch.
        """
        self.model.eval()
        all_embeddings = []
        
        for batch in tqdm(self.inference_loader, desc="Generating Embeddings", leave=False):
            batch = batch.to(self.device)
                        
            out = self.model(
                edge_index=batch.edge_index,
                edge_type=batch.edge_type,
                node_ids=batch.n_id,
                item_feats=self.item_feats
            )
            num_target_nodes = batch.batch_size
            target_embs = out[:num_target_nodes]
            all_embeddings.append(target_embs.cpu()) # upload to cpu to free GPU memory

        return torch.cat(all_embeddings, dim=0)
    
    @torch.no_grad()
    def evaluate_full_ranking(self, batch_size_eval=100):
        """
        Evaluate ranking metrics on all the embeddings.
        1. Obtain all the embeddings (on CPU).
        2. Upload all item embeddings on GPU.
        3. Upload user embeddings on GPU by batch.
        4. Calculate metrics.
        """
        self.model.eval()
        # generate embeddings
        all_embs_cpu = self.get_all_embeddings()
        num_users = self.model.num_users
        
        item_embs = all_embs_cpu[num_users:].to(self.device)
        
        # identify validation users
        val_users = np.unique(self.val_ground_truth.nonzero()[0])
        n_val = len(val_users)

        metrics_accum = {}
        for name in self.evaluator.metric_names:
            for k in self.evaluator.k_list:
                metrics_accum[f'{name}@{k}'] = 0.0
        
        # chunked evaluation loop
        num_batches = (n_val + batch_size_eval - 1) // batch_size_eval
        
        for i in tqdm(range(num_batches), desc="Evaluating Ranking", leave=False):
            start = i * batch_size_eval
            end = min((i + 1) * batch_size_eval, n_val)
            batch_user_ids = val_users[start:end]
            batch_user_embs = all_embs_cpu[batch_user_ids].to(self.device)

            scores = torch.matmul(batch_user_embs, item_embs.t())
            batch_gt_sparse = self.val_ground_truth[batch_user_ids]
            
            batch_metrics = self.evaluator.evaluate_batch(
                scores=scores,
                ground_truth_batch=batch_gt_sparse
            )
            
            current_batch_size = len(batch_user_ids)
            for k, v in batch_metrics.items():
                metrics_accum[k] += v * current_batch_size

        final_metrics = {k: v / n_val for k, v in metrics_accum.items()}
        
        self.logger.log_metrics({f'Val/{k}': v for k, v in final_metrics.items()}, self.step)
        return final_metrics
