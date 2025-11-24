import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import os
import logging
import random

from src.datasets.preprocessor import YambdaPreprocessor
from src.datasets.dataloader import create_pyg_data, create_loaders
from src.logger.logger import ExperimentLogger
from src.trainer.trainer import MBGCNTrainer
from src.metrics.ranking_metrics import RankingEvaluator
from src.loss.bpr_loss import BPRLoss
from src.model.mbgcn import MBGCN

# configure logger
log = logging.getLogger(__name__)

def seed_everything(seed: int):
    """Fix seed for experiments' reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_csr_matrix(df, user_map, item_map, num_users, num_items) -> sp.csr_matrix:
    """
    Creates sparse CSR matrix (users x items).
    """
    df_filtered = df[df['uid'].isin(user_map) & df['item_id'].isin(item_map)].copy()
    
    if df_filtered.empty:
        log.warning("Warning: Interaction matrix is empty after filtering!")
        return sp.csr_matrix((num_users, num_items), dtype=np.float32)

    # map id to idx
    user_indices = df_filtered['uid'].map(user_map).values
    item_indices = df_filtered['item_id'].map(item_map).values
    data = np.ones_like(user_indices, dtype=np.float32)
    
    return sp.csr_matrix((data, (user_indices, item_indices)), shape=(num_users, num_items))

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    # initialize environment
    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    
    # initialize logger
    logger = ExperimentLogger(cfg)
    log.info(f"Experiment: {cfg.experiment_name} on {device}")

    # proprocess data
    data_path = hydra.utils.to_absolute_path(cfg.dataset.event_path)
    embedding_path = None
    if 'embedding_path' in cfg.dataset and cfg.dataset.embedding_path:
        embedding_path = hydra.utils.to_absolute_path(cfg.dataset.embedding_path)
        log.info(f"Embedding path set to: {embedding_path}")
    log.info(f"Loading data from {data_path}...")
    
    preprocessor = YambdaPreprocessor(event_path=data_path, embedding_path=embedding_path)
    preprocessor.load_data()
    preprocessor.filter_interaction_types(cfg.dataset.get('interaction_types', ['listen', 'like']))
    if cfg.dataset.get('k_core', 0) > 0:
        preprocessor.apply_k_core(k=cfg.dataset.k_core)
    
    
    # perform temporal split
    train_df, val_df, test_df = preprocessor.temporal_split(
        test_days=cfg.dataset.get('test_days', 1),
        val_days=cfg.dataset.get('val_days', 1)
    )
    
    # builds graph
    data, user_map, item_map, num_users, num_items = create_pyg_data(train_df, val_df, test_df)
    log.info(f"Graph stats: Users={num_users}, Items={num_items}, Edges={data.num_edges}")
    
    # create ground truth matrices
    log.info("Building Ground Truth matrices...")
    train_gt = build_csr_matrix(train_df, user_map, item_map, num_users, num_items)
    val_gt = build_csr_matrix(val_df, user_map, item_map, num_users, num_items)
    
    # create loaders
    loaders = create_loaders(
        data, train_df, val_df, test_df,
        user_map, item_map,
        batch_size=cfg.dataloader.batch_size,
        num_neighbors=list(cfg.dataloader.num_neighbors),
        neg_sampling_ratio=cfg.dataloader.get('neg_sampling_ratio', 1.0)
    )
    
    # initialize a model
    item_feats = None
    use_pretrained = False
    input_item_dim = cfg.model.item_feat_dim

    if preprocessor.embeddings is not None:
        log.info("Processing pretrained item embeddings...")
        if 'normalized_embed' not in preprocessor.embeddings.columns:
            log.warning("Embeddings dataframe does not have 'normalized_embed' column. Skipping.")
        else:
            sample_emb = preprocessor.embeddings['normalized_embed'].iloc[0]
            embed_dim = len(sample_emb)
            
            item_feats_tensor = torch.zeros((num_items, embed_dim), dtype=torch.float32)
            valid_emb_df = preprocessor.embeddings[
                preprocessor.embeddings['item_id'].isin(item_map.keys())
            ].copy()
            
            if not valid_emb_df.empty:
                valid_emb_df['graph_idx'] = valid_emb_df['item_id'].map(item_map)
                indices = valid_emb_df['graph_idx'].values
                vectors = np.stack(valid_emb_df['embed'].values)
                item_feats_tensor[indices] = torch.from_numpy(vectors).float()
                item_feats = item_feats_tensor
                input_item_dim = embed_dim
                use_pretrained = True
                log.info(f"Successfully aligned embeddings for {len(indices)}/{num_items} items. Dim: {embed_dim}")
            else:
                log.warning("No intersection between graph items and loaded embeddings!")
    
    model = MBGCN(
        num_users=num_users,
        num_items=num_items,
        num_behaviours=1,
        item_feat_dim=input_item_dim,
        node_emb_dim=cfg.model.node_emb_dim,
        user_emb_dim=cfg.model.user_emb_dim,
        behaviour_hidden_dim=cfg.model.behaviour_hidden_dim,
        n_gnn_layers=cfg.model.n_gnn_layers,
        fusion=cfg.model.fusion,
        dropout=cfg.model.dropout,
        use_pretrained_item_feats=use_pretrained
    ).to(device)
    
    # training components
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.get('weight_decay', 0.0)
    )
    criterion = BPRLoss()
    
    # initialize ranking evaluator
    evaluator = RankingEvaluator(
        k_list=cfg.metrics.get('k_list', [10, 20, 50]),
        metric_names=cfg.metrics.get('names', ['ndcg', 'recall'])
    )
    
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
    
    # training loop
    epochs = cfg.trainer.epochs
    best_metric = 0.0
    best_epoch = -1
    # main metric for saving the checkpoint
    primary_metric = f"val/ndcg@{cfg.metrics.get('k_list', [10, 20])[0]}"
    
    ckpt_dir = os.getcwd()
    log.info(f"Checkpoints will be saved to: {ckpt_dir}")

    log.info("Starting Training Loop...")
    
    for epoch in range(epochs):
        # train
        train_loss = trainer.train_epoch(epoch)
        log.info(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f}")
        
        # validation
        val_interval = cfg.trainer.get('val_interval', 1)
        
        if (epoch + 1) % val_interval == 0:
            log.info("Running evaluation...")
            
            metrics = trainer.evaluate_full_ranking()
            
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k,v in metrics.items()])
            log.info(f"Epoch {epoch:02d} | {metrics_str}")
            
            target_metric_key = primary_metric.split('/')[-1]
            current_val = metrics.get(target_metric_key, 0.0)
            
            if current_val > best_metric:
                best_metric = current_val
                best_epoch = epoch
                
                ckpt_path = os.path.join(ckpt_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': OmegaConf.to_container(cfg),
                    'best_metric': best_metric
                }, ckpt_path)
                log.info(f"New best model saved! ({target_metric_key}: {best_metric:.4f})")

    log.info(f"Training finished. Best epoch: {best_epoch} with {primary_metric.split('/')[-1]}: {best_metric:.4f}")
    logger.close()

if __name__ == "__main__":
    main()
