from typing import Literal
import pandas as pd
from collections import defaultdict
import numpy as np
from Application.data import YambdaDataset, dataset
import scipy.sparse as sp
import heapq


class RecommendationSystem:
    def __init__(self, dataset: YambdaDataset):
        self.dataset = dataset
        self.likes = dataset.interaction("likes")
        self.likes_df = self.likes.to_pandas()
        self._build_user_item_matrix(self.likes_df)
        
    
    def _build_user_item_matrix(self, df):
        du = df[['item_id', 'uid']].drop_duplicates()

        item_codes, item_index = pd.factorize(du['item_id'], sort=False)
        user_codes, user_index = pd.factorize(du['uid'], sort=False)

        X = sp.csr_matrix(
            (np.ones(len(du), dtype=np.uint8), (item_codes, user_codes)),
            shape=(len(item_index), len(user_index))
        )

        item_id_to_idx = dict(zip(item_index, np.arange(len(item_index))))
        idx_to_item_id = np.array(item_index)
        item_pop = np.asarray(X.getnnz(axis=1), dtype=np.int32) # | U_i | 
        
        self.X = X
        self.item_pop = item_pop
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = idx_to_item_id
    
    def _nearest_items_topk_jaccard_sparse(self, item_original_id, top_k: int = 100):
        i = self.item_id_to_idx.get(item_original_id, None)
        if i is None or self.item_pop[i] == 0:
            return []

        intersection = self.X[i].dot(self.X.T).toarray().ravel()
        union = self.item_pop[i] + self.item_pop - intersection
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=(union > 0))
        similarity[i] = 0

        k = min(top_k, similarity.size - 1)
        idx = np.argpartition(similarity, -k)[-k:]
        idx = idx[np.argsort(-similarity[idx])]
        similarities_chosen = similarity[idx]

        return [(int(self.idx_to_item_id[j]), s) for j, s in zip(idx, similarities_chosen)]
    
    def _build_user_items(self, user_id):
        user_items = set(
            self.likes_df.loc[self.likes_df['uid'] == user_id, 'item_id'].unique()
        )
        return user_items
    
    def _get_item_rank(self, item_original_id, item_nearest_elements, user_items):
        num = 0
        denom = 0
        for it_id, sim in item_nearest_elements:
            if it_id == item_original_id:
                continue
            if it_id in user_items:
                num += sim
            denom += abs(sim)

        if denom == 0:
            return 0.0

        return num / denom
    
    def recommend_items_to_user(self, user_id, neighbors_per_liked=50, top_n=100, prebuilt=None, user_items=None):

        if not user_items:
            user_items = set(self.likes_df.loc[self.likes_df['uid'] == user_id, 'item_id'].unique())
            if not user_items:
                return []
        

        seeds = np.array(list(user_items), dtype=np.int64)
        scores = defaultdict(float)

        for seed in seeds:
            neigh = self._nearest_items_topk_jaccard_sparse(seed, neighbors_per_liked)
            for nid, s in neigh:
                if nid in user_items or nid == seed:
                    continue
                scores[nid] += s
                
        if not scores:
            return []

        top = heapq.nlargest(top_n, scores.items(), key=lambda kv: kv[1])
        return top
        
        
            

recommendation_system = RecommendationSystem(dataset)