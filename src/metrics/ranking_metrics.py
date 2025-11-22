import torch
from typing import List, Dict

class RankingEvaluator:
    def __init__(self, k_list: List[int], metric_names: List[str]):
        self.k_list = sorted(k_list)
        self.metric_names = [m.lower() for m in metric_names]
        self.max_k = self.k_list[-1]

    def _calculate_ndcg_score(self, hits: torch.Tensor, num_positives: torch.Tensor, k: int) -> float:
        device = hits.device
        batch_size = hits.size(0)

        discounts = 1.0 / torch.log2(torch.arange(k, device=device, dtype=torch.float) + 2.0)
        dcg = (hits * discounts).sum(dim=1)
        
        positions = torch.arange(k, device=device).unsqueeze(0).expand(batch_size, -1)

        ideal_mask = (positions < num_positives.unsqueeze(1)).float()
        idcg = (ideal_mask * discounts).sum(dim=1)
        idcg[idcg == 0] = 1.0
        return (dcg / idcg).mean().item()

    def evaluate_full_ranking(self, 
                              user_emb: torch.Tensor, 
                              ground_truth_mask: torch.Tensor, 
                              all_item_emb: torch.Tensor,
                              history_mask: torch.Tensor = None) -> Dict[str, float]:
        """
        Args:
            user_emb: user embeddigns (batch_size, embedding_dim).
            ground_truth_mask: mask of item relevancy (batch_size, num_items).
            all_item_emb: item embeddings (num_items, embedding_dim).
        """
        # calculate scores between users and items
        scores = torch.matmul(user_emb, all_item_emb.t())

        # select indices of items with maximal scores for each user 
        _, indices = torch.topk(scores, k=self.max_k, dim=1)
        # calculate hit matrix, 1 - if predicted item is relevant, 0 - otherwise
        hits = torch.gather(ground_truth_mask, 1, indices)
        # calculate number of total relevant items for each user
        num_relevant = ground_truth_mask.sum(dim=1)

        results = {}        
        for k in self.k_list:
            hits_k = hits[:, :k]
            
            if 'recall' in self.metric_names:
                rec = hits_k.sum(dim=1) / num_relevant.clamp(min=1.0)
                results[f'recall@{k}'] = rec.mean().item()
                
            if 'ndcg' in self.metric_names:
                results[f'ndcg@{k}'] = self._calculate_ndcg_score(hits_k, num_relevant, k)
                
        return results
