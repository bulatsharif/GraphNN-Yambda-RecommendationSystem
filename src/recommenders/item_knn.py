import numpy as np
import scipy.sparse as sp
import pandas as pd
import heapq
from tqdm.auto import tqdm
from collections import defaultdict
from typing import List, Dict, Tuple
from functools import lru_cache # Import lru_cache
from .BaseRecommender import BaseRecommender

class ItemKNN(BaseRecommender):
    def __init__(self, neighbors_per_liked: int = 50):
        self.neighbors_per_liked = neighbors_per_liked
        self.X = None
        self.Xt = None # Store transposed matrix
        self.item_pop = None
        self.item_id_to_idx = None
        self.idx_to_item_id = None
        self.user_history = defaultdict(set)

    def fit(self, df: pd.DataFrame, test_df=None):
        """Builds the User-Item matrix and pre-computes statistics."""
        # Store user history for fast lookup
        for uid, group in df.groupby('uid'):
            self.user_history[uid] = set(group['item_id'].tolist())

        du = df[['item_id', 'uid']].drop_duplicates()

        # Create mappings
        item_codes, item_index = pd.factorize(du['item_id'], sort=False)
        user_codes, user_index = pd.factorize(du['uid'], sort=False)

        # Build Sparse Matrix
        self.X = sp.csr_matrix(
            (np.ones(len(du), dtype=np.uint8), (item_codes, user_codes)),
            shape=(len(item_index), len(user_index))
        )
        
        # [OPTIMIZATION 1] Pre-compute and cache the transposed matrix (CSC format is efficient for columns)
        self.Xt = self.X.T.tocsr()

        self.item_id_to_idx = dict(zip(item_index, np.arange(len(item_index))))
        self.idx_to_item_id = np.array(item_index)
        self.item_pop = np.asarray(self.X.getnnz(axis=1), dtype=np.int32)

    # [OPTIMIZATION 2] Cache results for recently queried items
    @lru_cache(maxsize=10000) 
    def _get_nearest_items(self, item_original_id, top_k: int) -> List[Tuple[int, float]]:
        i = self.item_id_to_idx.get(item_original_id, None)
        if i is None or self.item_pop[i] == 0:
            return []

        # [OPTIMIZATION 1] Use pre-computed Xt
        intersection = self.X[i].dot(self.Xt).toarray().ravel()
        union = self.item_pop[i] + self.item_pop - intersection
        
        # Avoid division by zero
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=(union > 0))
        similarity[i] = 0  # Remove self

        k = min(top_k, similarity.size - 1)
        idx = np.argpartition(similarity, -k)[-k:]
        idx = idx[np.argsort(-similarity[idx])]
        similarities_chosen = similarity[idx]

        return [(int(self.idx_to_item_id[j]), s) for j, s in zip(idx, similarities_chosen)]

    def recommend(self, user_ids: List[int], k: int, history_filter: Dict[int, List[int]] = None) -> Dict[int, List[int]]:
        recommendations = {}
        
        # Clear cache before large batch prediction if necessary, or let it warm up
        self._get_nearest_items.cache_clear() 
        
        for i, uid in tqdm(enumerate(user_ids), total=len(user_ids), desc="Generating recommendations"):
            # ... (Rest of logic remains the same) ...
            user_items = set()
            if history_filter and uid in history_filter:
                user_items = set(history_filter[uid])
            elif uid in self.user_history:
                user_items = self.user_history[uid]
            
            if not user_items:
                recommendations[uid] = []
                continue

            seeds = list(user_items)
            scores = defaultdict(float)

            for seed in seeds:
                neighbors = self._get_nearest_items(seed, self.neighbors_per_liked)
                for nid, s in neighbors:
                    if nid in user_items or nid == seed:
                        continue
                    scores[nid] += s
            
            if not scores:
                recommendations[uid] = []
            else:
                top_items = heapq.nlargest(k, scores.items(), key=lambda kv: kv[1])
                recommendations[uid] = [item for item, score in top_items]
        
        return recommendations