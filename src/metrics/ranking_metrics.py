import torch
import numpy as np
from typing import List, Dict
import scipy.sparse as sp

class RankingEvaluator:
    def __init__(self, k_list: List[int], metric_names: List[str]):
        self.k_list = sorted(k_list)
        self.metric_names = [m.lower() for m in metric_names]
        self.max_k = self.k_list[-1]

    def _calculate_ndcg(self, hits: np.ndarray, num_relevant: np.ndarray, k: int) -> float:
        discounts = 1.0 / np.log2(np.arange(k) + 2.0)
        dcg = (hits * discounts).sum(axis=1)
        
        top_k_range = np.arange(k)
        ideal_hits = (top_k_range[None, :] < num_relevant[:, None]).astype(np.float32)
        idcg = (ideal_hits * discounts).sum(axis=1)
        
        idcg[idcg == 0] = 1.0
        return (dcg / idcg).mean()

    def evaluate_batch(self, 
                       scores: torch.Tensor, 
                       ground_truth_batch: sp.csr_matrix) -> Dict[str, float]:
        _, topk_indices = torch.topk(scores, k=self.max_k, dim=1)
        topk_indices = topk_indices.detach().cpu().numpy()

        batch_size = topk_indices.shape[0]
        hits = np.zeros((batch_size, self.max_k), dtype=np.float32)
        num_relevant = np.diff(ground_truth_batch.indptr)

        for i in range(batch_size):
            start, end = ground_truth_batch.indptr[i], ground_truth_batch.indptr[i+1]
            true_items = ground_truth_batch.indices[start:end]
            
            if len(true_items) == 0:
                continue
            
            hits[i] = np.isin(topk_indices[i], true_items).astype(np.float32)

        results = {}
        for k in self.k_list:
            hits_k = hits[:, :k]
            
            if 'recall' in self.metric_names:
                relevant_clamp = np.maximum(num_relevant, 1.0)
                recall = hits_k.sum(axis=1) / relevant_clamp
                results[f'recall@{k}'] = recall.mean()
            
            if 'ndcg' in self.metric_names:
                results[f'ndcg@{k}'] = self._calculate_ndcg(hits_k, num_relevant, k)
        return results
