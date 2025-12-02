import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import scipy.sparse as sp
import numpy as np
import os
import random
import gc

from src.datasets.preprocessor import YambdaPreprocessor
from src.datasets.dataloader import create_pyg_data, create_loaders
from src.logger.logger import ExperimentLogger
from src.trainer.trainer import MBGCNTrainer
from src.trainer.lightgcn_trainer import LightGCNTrainer
from src.metrics.ranking_metrics import RankingEvaluator
from src.loss.bpr_loss import BPRLoss
from src.model.mbgcn import MBGCN
from src.model.lightgcn import LightGCN

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_csr_matrix(df, user_map, item_map, num_users, num_items, target_event) -> sp.csr_matrix:
    mask = (df['event_type'] == target_event) & \
           (df['uid'].isin(user_map)) & \
           (df['item_id'].isin(item_map))
    df_filtered = df[mask]
    
    if df_filtered.empty:
        print(f"Warning: GT matrix for {target_event} is empty!")
        return sp.csr_matrix((num_users, num_items), dtype=np.float32)

    user_indices = df_filtered['uid'].map(user_map).values
    item_indices = df_filtered['item_id'].map(item_map).values
    data = np.ones(len(user_indices), dtype=np.float32)
    
    return sp.csr_matrix((data, (user_indices, item_indices)), shape=(num_users, num_items))

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    logger = ExperimentLogger(cfg)
    sampling_ratio = cfg.dataset.get('train_sampling_ratio', 1.0)
    print(f"Experiment: {cfg.experiment_name} | Target: {cfg.target_event} | Sampling: {sampling_ratio}")

    data_dir = hydra.utils.to_absolute_path(cfg.dataset.data_dir)
    embedding_path = None
    if 'embedding_path' in cfg.dataset and cfg.dataset.embedding_path:
        embedding_path = hydra.utils.to_absolute_path(cfg.dataset.embedding_path)

    preprocessor = YambdaPreprocessor(data_dir=data_dir, embedding_path=embedding_path)
    
    interaction_types = list(cfg.dataset.interaction_types)
    print(f"Loading data with interaction types: {interaction_types}")
    train_df, val_df, test_df = preprocessor.load_data(interaction_types, sampling_ratio)
    
    data, user_map, item_map, num_users, num_items = create_pyg_data(
        train_df, val_df, test_df, interaction_types
    )
    print(f"Graph stats: Users={num_users}, Items={num_items}, Edges={data.num_edges}")
    
    try:
        target_event_idx = interaction_types.index(cfg.target_event)
    except ValueError:
        raise ValueError(f"Target event {cfg.target_event} not in interaction_types {interaction_types}")

    print(f"Building Ground Truth matrices for target: {cfg.target_event} (Idx: {target_event_idx})...")
    train_gt = build_csr_matrix(train_df, user_map, item_map, num_users, num_items, cfg.target_event)
    val_gt = build_csr_matrix(val_df, user_map, item_map, num_users, num_items, cfg.target_event)
    
    loaders = create_loaders(
        data=data,
        train_df=train_df,
        user_map=user_map,
        item_map=item_map,
        batch_size=cfg.dataloader.batch_size,
        num_neighbors=list(cfg.dataloader.num_neighbors),
        target_event=cfg.target_event
    )
    
    print("Cleaning up raw dataframes to save RAM...")
    del train_df, val_df, test_df
    gc.collect()
    
    item_feats = None
    use_pretrained = False
    input_item_dim = cfg.model.get('item_feat_dim', 128)

    if preprocessor.embeddings is not None:
        print("Processing pretrained item embeddings...")
        valid_emb_df = preprocessor.embeddings[
            preprocessor.embeddings['item_id'].isin(item_map.keys())
        ].copy()
        
        if not valid_emb_df.empty:
            sample = valid_emb_df['normalized_embed'].iloc[0]
            embed_dim = len(sample) if hasattr(sample, '__len__') else 128
            
            item_feats_tensor = torch.zeros((num_items, embed_dim), dtype=torch.float32)
            valid_emb_df['graph_idx'] = valid_emb_df['item_id'].map(item_map)
            
            indices = torch.tensor(valid_emb_df['graph_idx'].values, dtype=torch.long)
            vectors_np = np.stack(valid_emb_df['normalized_embed'].values)
            vectors = torch.from_numpy(vectors_np).float()
            
            item_feats_tensor[indices] = vectors
            item_feats = item_feats_tensor
            input_item_dim = embed_dim
            use_pretrained = True
            print(f"Aligned embeddings: {len(indices)}/{num_items}. Dim: {embed_dim}")
            
            del preprocessor.embeddings, valid_emb_df
            gc.collect()

    # Model Selection Logic
    target_class = cfg.model._target_
    print(f"Initializing Model: {target_class}")
    
    if "MBGCN" in target_class:
        model = MBGCN(
            num_users=num_users,
            num_items=num_items,
            num_behaviours=data.num_behaviours,
            item_feat_dim=input_item_dim,
            node_emb_dim=cfg.model.node_emb_dim,
            user_emb_dim=cfg.model.user_emb_dim,
            behaviour_hidden_dim=cfg.model.behaviour_hidden_dim,
            n_gnn_layers=cfg.model.n_gnn_layers,
            fusion=cfg.model.fusion,
            dropout=cfg.model.dropout,
            use_pretrained_item_feats=use_pretrained
        ).to(device)
    elif "LightGCN" in target_class:
        model = LightGCN(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=cfg.model.embedding_dim,
            n_layers=cfg.model.n_layers,
            use_pretrained_item_feats=use_pretrained,
            item_feat_dim=input_item_dim
        ).to(device)
    else:
        raise ValueError(f"Unknown model target: {target_class}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    criterion = BPRLoss()
    
    evaluator = RankingEvaluator(
        k_list=cfg.metrics.get('k_list', [10, 100]),
        metric_names=cfg.metrics.get('names', ['ndcg', 'recall'])
    )
    
    # Trainer Selection Logic
    if "MBGCN" in target_class:
        trainer = MBGCNTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loaders=loaders,
            full_data=data,
            evaluator=evaluator,
            train_ground_truth=train_gt,
            val_ground_truth=val_gt,
            logger=logger,
            device=device,
            item_feats=item_feats
        )
    elif "LightGCN" in target_class:
        trainer = LightGCNTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            loaders=loaders,
            full_data=data,
            evaluator=evaluator,
            train_ground_truth=train_gt,
            val_ground_truth=val_gt,
            logger=logger,
            device=device,
            target_event_idx=target_event_idx,
            item_feats=item_feats
        )
    
    epochs = cfg.trainer.epochs
    best_metric = 0.0
    primary_metric_key = f"ndcg@{cfg.metrics.get('k_list', [10])[0]}"
    
    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = trainer.train_epoch(epoch)
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}")
        
        if (epoch + 1) % cfg.trainer.get('val_interval', 1) == 0:
            metrics = trainer.evaluate_full_ranking()
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k,v in metrics.items()])
            print(f"Epoch {epoch:02d} | {metrics_str}")
            
            current_val = metrics.get(primary_metric_key, 0.0)
            if current_val > best_metric:
                best_metric = current_val
                save_path = "best_model.pt" 
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'best_metric': best_metric
                }, save_path)
                print(f"New best model saved! ({primary_metric_key}: {best_metric:.4f})")
    logger.close()

if __name__ == "__main__":
    main()
