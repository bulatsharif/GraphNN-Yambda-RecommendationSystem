import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

class YambdaSequenceDataset(Dataset):
    def __init__(self, df, max_len=20):
        self.max_len = max_len
        self.user_sequences = df.groupby('uid')['item_id'].apply(list).values
        
    def __len__(self):
        return len(self.user_sequences)

    def __getitem__(self, idx):
        seq = self.user_sequences[idx]
        
        if len(seq) < 2:
            x_seq = seq
            y_item = seq[-1]
        else:
            x_seq = seq[:-1]
            y_item = seq[-1]
            
        if len(x_seq) > self.max_len:
            x_seq = x_seq[-self.max_len:]
            
        return torch.tensor(x_seq, dtype=torch.long), torch.tensor(y_item, dtype=torch.long)

def collate_fn(batch):
    x_list, y_list = zip(*batch)
    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0)
    y_tensor = torch.stack(y_list)
    return x_padded, y_tensor

def create_dataloaders(train_df, val_df, test_df, batch_size=32):
    train_loader = DataLoader(YambdaSequenceDataset(train_df), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(YambdaSequenceDataset(val_df), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(YambdaSequenceDataset(test_df), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader, test_loader
