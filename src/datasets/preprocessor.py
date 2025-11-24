import pandas as pd
import os
from typing import List, Tuple

class YambdaPreprocessor:
    def __init__(self, data_dir: str, embedding_path: str = None):
        self.data_dir = data_dir
        self.embedding_path = embedding_path
        self.embeddings = None

    def _convert_and_filter(self, df: pd.DataFrame, allowed_types: List[str]) -> pd.DataFrame:
        """
        1. Converts 'listen' to 'listen+' if played_ratio_pct > 50.
        2. Filters only allowed event types.
        """
        df = df.copy()
        
        listen_mask = (df['event_type'] == 'listen')
        played_ratio = df['played_ratio_pct'].fillna(0)
        valid_listen_mask = listen_mask & (played_ratio > 50)
        
        df.loc[valid_listen_mask, 'event_type'] = 'listen+'
        
        df_filtered = df[df['event_type'].isin(allowed_types)].copy()
        return df_filtered

    def load_data(self, interaction_types: List[str], train_sampling_ratio: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads data and optionally samples the training set.
        """
        print(f"Loading raw data from {self.data_dir}...")
        train_df = pd.read_parquet(os.path.join(self.data_dir, 'multi_event_train.parquet'))
        val_df = pd.read_parquet(os.path.join(self.data_dir, 'multi_event_val.parquet'))
        test_df = pd.read_parquet(os.path.join(self.data_dir, 'multi_event_test.parquet'))

        if self.embedding_path and os.path.exists(self.embedding_path):
            print(f"Loading embeddings from {self.embedding_path}")
            self.embeddings = pd.read_parquet(self.embedding_path)
            
        print(f"Filtering interactions. Allowed: {interaction_types}")
        train_df = self._convert_and_filter(train_df, interaction_types)
        val_df = self._convert_and_filter(val_df, interaction_types)
        test_df = self._convert_and_filter(test_df, interaction_types)
        
        if train_sampling_ratio < 1.0 and train_sampling_ratio > 0.0:
            original_size = len(train_df)
            train_df = train_df.sample(frac=train_sampling_ratio, random_state=42)
            print(f"SAMPLED Train Data: {train_sampling_ratio*100}% | {original_size} -> {len(train_df)} rows")
        
        return train_df, val_df, test_df
    
    def sync_embeddings(self, all_item_ids):
        """Filters embeddings to keep only those that present in the graph."""
        if self.embeddings is not None:
            self.embeddings = self.embeddings[self.embeddings['item_id'].isin(all_item_ids)]
