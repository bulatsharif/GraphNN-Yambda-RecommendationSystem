import torch

def calc_recall_at_k(scores, labels, k):
    """
    Calculate mean Recall@k for a batch of queries.
    
    Args:
        scores (torch.Tensor): Model's predictions (batch_size, num_items)
        labels (torch.Tensor): Ground Truth (batch_size, num_items)
        k (int): cutoff threshold
    """
    # find indices of top-k maximal scores for each query in a batch, shape (batch_size, k)
    _, indices = torch.topk(scores, k, dim=1)
    # calculate the row indices for advanced indexing of shape (batch_size, k)
    row_indices = torch.arange(scores.shape[0]).view(-1, 1).expand_as(indices)
    # find labels corresponding to prediction indices, shape (batch_size, k)
    topk_labels = labels[row_indices, indices]
    # calculate hits (count of relevant items among predicted)
    hits = topk_labels.sum(dim=1)
    # calculate the total number of relevant items for user
    total_relevant = labels.sum(dim=1)
    # calculate recall@k
    epsilon = 1e-9
    recall = hits / (total_relevant + epsilon)

    return recall.mean()

def calc_ndcg_at_k(scores, labels, k):
    """
    Calculate mean NDCG@k for a batch of queries.
    
    Args:
        scores (torch.Tensor): Model's predictions (batch_size, num_items)
        labels (torch.Tensor): Ground Truth (batch_size, num_items)
        k (int): cutoff threshold
    """
    # find indices of top-k maximal scores for each query in a batch, shape (batch_size, k)
    _, indices = torch.topk(scores, k, dim=1)
    # calculate the row indices for advanced indexing of shape (batch_size, k)
    row_indices = torch.arange(scores.shape[0]).view(-1, 1).expand_as(indices)
    # find labels corresponding to prediction indices, shape (batch_size, k)
    topk_relevance = labels[row_indices, indices]
    # calculate logariphmical discounts
    positions = torch.arange(2, k + 2, device=scores.device, dtype=torch.float32)
    discounts = torch.log2(positions)
    # calculate discounted cumulative gain
    dcg = (topk_relevance / discounts).sum(dim=1)
    
    # find sorted labels (desc) to calculate ideal dcg
    sorted_labels, _ = torch.topk(labels, k, dim=1) 
    idcg = (sorted_labels / discounts).sum(dim=1)
    
    # calculate normalized dcg
    epsilon = 1e-9
    ndcg = dcg / (idcg + epsilon)
    ndcg[idcg == 0] = 0
    
    return ndcg.mean()
