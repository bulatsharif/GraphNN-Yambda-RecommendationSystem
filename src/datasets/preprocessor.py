import pandas as pd
import os

class YambdaPreprocessor:
    def __init__(self, event_path, embedding_path=None):
        self.event_path = event_path
        self.embedding_path = embedding_path
        self.df = None
        self.embeddings = None
        
    def load_data(self):
        self.df = pd.read_parquet(self.event_path, engine='fastparquet')
        
        if self.embedding_path:
            self.embeddings = pd.read_parquet(self.embedding_path)

    def filter_interaction_types(self, event_types):
        """
        Filters the dataset to only include specific interaction types.
        Handles special logic for 'listen+' (listen with played_ratio_pct > 50).
        """
        # 1. Handle 'listen+' Logic: Virtual Event Creation
        # We only perform this transformation if 'listen+' is explicitly requested.
        # This prevents accidental data loss if the user just wants standard 'listen' events.
        
        if 'listen+' in event_types:
            # Identify qualifying rows
            mask_high_quality = (self.df['event_type'] == 'listen') & (self.df['played_ratio_pct'] > 50)
            
            # Rename them to 'listen+'
            # Note: Rows with <= 50 remain as 'listen'
            self.df.loc[mask_high_quality, 'event_type'] = 'listen+'

        # 2. Filter
        self.df = self.df[self.df['event_type'].isin(event_types)].copy()

        # Sync embeddings if they exist (Remove embeddings for items that were filtered out)
        if self.embeddings is not None:
            remaining_items = set(self.df['item_id'].unique())
            self.embeddings = self.embeddings[self.embeddings['item_id'].isin(remaining_items)]

    def apply_k_core(self, k=5):
        """
        Iteratively removes users and items with fewer than k interactions
        until the condition is met for all remaining data.
        """
        df = self.df.copy()
            
        iteration = 0
        
        while True:
            iteration += 1
            # Count occurrences
            user_counts = df['uid'].value_counts()
            item_counts = df['item_id'].value_counts()
            
            # Find valid entities
            valid_users = user_counts[user_counts >= k].index
            valid_items = item_counts[item_counts >= k].index
            
            # Filter
            new_df = df[
                (df['uid'].isin(valid_users)) & 
                (df['item_id'].isin(valid_items))
            ]
            
            # Check convergence
            if len(new_df) == len(df) or df.empty:
                break
            
            df = new_df
        
        self.df = df
        
        # Sync embeddings if they exist (Remove embeddings for items that were filtered out)
        if self.embeddings is not None:
            remaining_items = set(self.df['item_id'].unique())
            self.embeddings = self.embeddings[self.embeddings['item_id'].isin(remaining_items)]

    def temporal_split(self, test_days=1, val_days=1, gap_minutes=30):
        """
        Applies temporal split based on the max timestamp in the data.
        """
        self.df = self.df.sort_values(by='timestamp', ascending=True)

        # Dynamic max based on data
        timestamp_max = self.df['timestamp'].max()
        secs_per_day = 60 * 60 * 24
        gap_seconds = gap_minutes * 60

        # Calculate thresholds backwards
        test_lo = timestamp_max - (test_days * secs_per_day)
        
        val_hi = test_lo - gap_seconds
        val_lo = val_hi - (val_days * secs_per_day)
        
        train_hi = val_lo - gap_seconds

        train = self.df[self.df['timestamp'] < train_hi]
        val = self.df[(self.df['timestamp'] >= val_lo) & (self.df['timestamp'] < val_hi)]
        test = self.df[self.df['timestamp'] >= test_lo]
        
        return train, val, test

    def save_processed(self, train, val, test, output_dir='processed_data'):
        os.makedirs(output_dir, exist_ok=True)
        train.to_parquet(f'{output_dir}/multi_event_train.parquet', index=False, compression='lz4')
        val.to_parquet(f'{output_dir}/multi_event_val.parquet', index=False, compression='lz4')
        test.to_parquet(f'{output_dir}/multi_event_test.parquet', index=False, compression='lz4')
        
        if self.embeddings is not None:
            if 'embed' in self.embeddings.columns:
                embeddings_compressed = self.embeddings.drop(columns=['embed'])
                embeddings_compressed.to_parquet(f'{output_dir}/embeddings.parquet', index=False, compression='lz4')
