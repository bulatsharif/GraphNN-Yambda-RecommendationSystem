import hydra
from omegaconf import DictConfig
from clearml import Task, Logger
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from src.recommenders.item_knn import ItemKNN

def compute_recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    rec_k = set(recommended[:k])
    return len(rec_k.intersection(relevant)) / len(relevant)

def evaluate_model(model, test_df, k, limit=None):
    """
    Evaluates the model on the test set.
    """
    # Group test data by user to get ground truth
    test_user_interactions = test_df.groupby('uid')['item_id'].apply(set).to_dict()
    test_users = list(test_user_interactions.keys())
    
    if limit:
        test_users = test_users[:limit]

    # Generate recommendations
    # Note: The model's fit() method already stored the training history internally, 
    # which it uses as the 'history_filter' implicitly for seeds.
    preds = model.recommend(test_users, k=k)
    
    recalls = []
    logger = Logger.current_logger()
    pbar = tqdm(test_users, desc="Calculating metrics")
    for idx, uid in enumerate(pbar, start=1):
        ground_truth = test_user_interactions[uid]
        recommendations = preds.get(uid, [])
        
        recall = compute_recall_at_k(recommendations, ground_truth, k)
        recalls.append(recall)
    return np.mean(recalls)

@hydra.main(version_base=None, config_path="../configs/eval", config_name="eval")
def main(cfg: DictConfig):
    # 1. Initialize ClearML
    task = Task.init(
        project_name=cfg.clearml.project_name, 
        task_name=cfg.clearml.task_name,
        tags=cfg.clearml.tags
    )
    task.connect(cfg) # Logs the Hydra config to ClearML

    print("Loading Data...")
    # Replace with your actual data loading logic
    train_df = pd.read_parquet(cfg.dataset.train_path)
    test_df = pd.read_parquet(cfg.dataset.test_path)
    
    print(f"Initializing {cfg.model.name}...")
    model = ItemKNN(neighbors_per_liked=cfg.model.neighbors_per_liked)
    
    print("Fitting model...")
    model.fit(train_df)
    
    print("Evaluating...")
    k = cfg.model.top_k
    limit = cfg.get("limit")
    recall_score = evaluate_model(model, test_df, k, limit=limit)
    
    print(f"Recall@{k}: {recall_score}")
    
    # Log metrics to ClearML
    Logger.current_logger().report_scalar(
        title="Evaluation", 
        series=f"Recall@{k}", 
        value=recall_score, 
        iteration=1
    )

if __name__ == "__main__":
    main()