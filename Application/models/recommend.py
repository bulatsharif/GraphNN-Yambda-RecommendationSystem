from typing import Literal
import pandas as pd
from collections import defaultdict
import numpy as np
from Application.data import YambdaDataset, dataset
import scipy.sparse as sp
import heapq


class RecommendationSystem:
    def __init__(self, dataset: YambdaDataset):
        # Store bound dataset instance
        self.dataset = dataset
        # Use "likes" as the primary implicit feedback signal for now.
        # (Future: generalize to multiple interaction types or parameterize the interaction name.)
        self.likes = dataset.interaction("likes")
        # Convert interaction events to a Pandas DataFrame for convenient manipulation
        self.likes_df = self.likes.to_pandas()
        # From raw (user,item) events we will construct a sparse item-user matrix suitable for item-based KNN style similarity.
        # We use a SciPy CSR matrix for memory efficiency and fast row operations.
        self._build_user_item_matrix(self.likes_df)
        
    
    def _build_user_item_matrix(self, df):
        # Keep only unique (item_id, uid) pairs; multiplicity of likes is ignored (binary implicit feedback)
        du = df[['item_id', 'uid']].drop_duplicates()

        # pd.factorize -> (codes, uniques). Codes are 0..n_unique-1. We factorize items & users separately to build compact indices.
        # Example: if item_codes[i] == 5 then item_index[5] is the original item_id. Same logic for users.
        item_codes, item_index = pd.factorize(du['item_id'], sort=False)
        user_codes, user_index = pd.factorize(du['uid'], sort=False)

        # Build item-user interaction matrix X (rows: items, cols: users) with 1 for an observed interaction.
        # Shape: (num_items, num_users). We use uint8 to minimize memory; values act as binary flags.
        X = sp.csr_matrix(
            (np.ones(len(du), dtype=np.uint8), (item_codes, user_codes)),
            shape=(len(item_index), len(user_index))
        )

        # Index mappings (both directions) for later translation between internal indices and original item_ids.
        item_id_to_idx = dict(zip(item_index, np.arange(len(item_index))))
        # Reverse lookup array: internal idx -> original item_id
        idx_to_item_id = np.array(item_index)
        # Item popularity: number of distinct users per item (|U_i|)
        item_pop = np.asarray(X.getnnz(axis=1), dtype=np.int32) # |U_i|
        
        # Cache constructed structures
        self.X = X
        self.item_pop = item_pop
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = idx_to_item_id
    
    def _nearest_items_topk_jaccard_sparse(self, item_original_id, top_k: int = 100):
        # Compute top-k most similar items (by Jaccard similarity over user sets) to the given item.
        # Jaccard(i,j) = |U_i ∩ U_j| / |U_i ∪ U_j| with implicit binary interactions.
        
        # Map original item id to internal index
        i = self.item_id_to_idx.get(item_original_id, None)
        if i is None or self.item_pop[i] == 0:
            return []

        # Intersection counts: row_i (1 x U) · X^T (U x I) -> dense 1 x I vector of co-occurrence counts |U_i ∩ U_j|.
        # Sparse optimization: cost ~ O(Σ_{u in U_i} deg(u)) — i.e. proportional to total interactions of users who liked item i.
        intersection = self.X[i].dot(self.X.T).toarray().ravel()
        # Union sizes via inclusion–exclusion: |A ∪ B| = |A| + |B| - |A ∩ B|
        union = self.item_pop[i] + self.item_pop - intersection
        # Safe division to avoid zero-division warnings
        similarity = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=np.float32), where=(union > 0))
        # Remove self-similarity
        similarity[i] = 0

        # Select top-k indices by similarity
        k = min(top_k, similarity.size - 1)
        # np.argpartition to get k largest in O(I) average (I = number of items)
        idx = np.argpartition(similarity, -k)[-k:]
        # Sort only the candidate subset: O(k log k)
        idx = idx[np.argsort(-similarity[idx])]
        # Retrieve similarity scores in order
        similarities_chosen = similarity[idx]

        # Complexity summary:
        #   Intersection computation: O(Σ_{u in U_i} deg(u))   (often far less than I*U)
        #   Argpartition: O(I)
        #   Sort top-k:  O(k log k)
        # Memory note: the dense similarity vector requires O(I) space due to toarray().
        return [(int(self.idx_to_item_id[j]), s) for j, s in zip(idx, similarities_chosen)]
    
    def _build_user_items(self, user_id):
        # Build the set of items the user has interacted with (binary implicit profile)
        user_items = set(
            self.likes_df.loc[self.likes_df['uid'] == user_id, 'item_id'].unique()
        )
        return user_items
    
    def _get_item_rank(self, item_original_id, item_nearest_elements, user_items):
        # Score an item by aggregating similarity weights of its neighbors present in the user profile.
        # Complexity: O(k) where k = len(item_nearest_elements). For small fixed k it is effectively constant.
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
        # Generate top-N recommended items for the given user via item-based neighborhood aggregation.
        # Steps:
        #   1. Obtain user profile (set of liked items)
        #   2. For each seed item s in profile, fetch top-k similar items
        #   3. Accumulate similarity scores for candidate items not already liked
        #   4. Return highest scoring unseen items
        if not user_items:
            user_items = set(self.likes_df.loc[self.likes_df['uid'] == user_id, 'item_id'].unique())
            if not user_items:
                return []
        
        # Seed items = user profile
        seeds = np.array(list(user_items), dtype=np.int64)
        # Accumulate candidate item scores
        scores = defaultdict(float)

        for seed in seeds:
            # For each seed item compute neighbors (see complexity in helper): O(Σ deg + I + k log k)
            neigh = self._nearest_items_topk_jaccard_sparse(seed, neighbors_per_liked)
            # Aggregate over k neighbors: O(k)
            for nid, s in neigh:
                if nid in user_items or nid == seed:
                    continue
                scores[nid] += s
        # Overall loop complexity ≈ O( Σ_{s in S} ( Σ_{u in U_s} deg(u) + I + k log k ) )
        # For sparse data and small |S| this is typically manageable; dominated by intersection computations.
                
        if not scores:
            return []

        # Select top-N candidates via heapq.nlargest: O(C log N) where C = number of candidate items (distinct neighbors across seeds)
        top = heapq.nlargest(top_n, scores.items(), key=lambda kv: kv[1])
        
        # Total recommendation complexity summarized:
        #   O( Σ_{s in S} ( Σ_{u in U_s} deg(u) + I + k log k ) + C log N )
        # Notation:
        #   S = set of seed items (user profile size)
        #   U_s = users who interacted with seed s
        #   deg(u) = number of items user u interacted with
        #   I = total number of items
        #   k = neighbors_per_liked
        #   C = number of unique candidate items produced
        return top
        
        
            

recommendation_system = RecommendationSystem(dataset)