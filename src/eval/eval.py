import hydra
from omegaconf import DictConfig, OmegaConf
from clearml import Task, Logger
import pandas as pd
import numpy as np
import os
from src.recommenders.BaseRecommender import BaseRecommender
from src.recommenders.item_knn import ItemKNN
from src.recommenders.mbgcn_recommender import MBGCNRecommender

def compute_recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = set(recommended[:k])
    return len(rec_k.intersection(relevant)) / len(relevant)

def evaluate_recommender(recommender: BaseRecommender, 
                         test_df: pd.DataFrame, 
                         train_df: pd.DataFrame, 
                         k: int):
    # 1. Build Ground Truth
    test_ground_truth = test_df.groupby('uid')['item_id'].apply(set).to_dict()
    test_users = list(test_ground_truth.keys())

    # 2. Build History Filter (Items seen in training)
    train_history = train_df.groupby('uid')['item_id'].apply(list).to_dict()

    # 3. Generate Recommendations
    print(f"Generating Top-{k} recommendations for {len(test_users)} users...")
    recommendations = recommender.recommend(
        user_ids=test_users, 
        k=k, 
        history_filter=train_history
    )

    # 4. Calculate Metrics
    print("Calculating Recall...")
    recalls = []
    for uid in test_users:
        if uid in recommendations:
            r = compute_recall_at_k(recommendations[uid], test_ground_truth[uid], k)
            recalls.append(r)
        else:
            recalls.append(0.0)
    
    return np.mean(recalls)

@hydra.main(version_base=None, config_path="../configs", config_name="eval/eval")
def main(cfg: DictConfig):
    # Unwrap config if it's nested under 'eval'
    if "eval" in cfg:
        cfg = cfg.eval
        
    recommender_name = cfg.recommender_name
    if cfg.get("clearml"):
        task = Task.init(
            project_name=cfg.clearml.project_name, 
            task_name=f"Eval-{recommender_name}", 
            tags=cfg.clearml.tags
        )
        task.connect(cfg)
    
    # 1. Load Data    
    train_path = cfg.dataset.train_file
    test_path = cfg.dataset.test_file
    print(f"Loading train data from {train_path} and test data from {test_path}...")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 2. Instantiate Recommender
    print(f"Instantiating recommender: {recommender_name}")
    
    if recommender_name == "ItemKNN":
        model = ItemKNN()
    elif recommender_name == "MBGCN":
        model_params = OmegaConf.to_container(cfg.model, resolve=True)
        model = MBGCNRecommender(
            model_params=model_params,
            item_features_path=cfg.item_features_path,
            checkpoint_path=cfg.model_path,
            device=cfg.device
        )
    else:
        raise ValueError(f"Unknown recommender: {recommender_name}")

    # 3. Fit Model
    print("Fitting model...")
    model.fit(train_df, test_df=test_df)

    # 4. Evaluate
    k = cfg.metrics.k
    print(f"Starting evaluation @ K={k}...")
    recall_score = evaluate_recommender(model, test_df, train_df, k)
    
    print(f"Final Recall@{k}: {recall_score:.6f}")

    if cfg.get("clearml"):
        Logger.current_logger().report_scalar(
            title="Evaluation", 
            series=f"Recall@{k}", 
            value=recall_score, 
            iteration=1
        )

if __name__ == "__main__":
    main()