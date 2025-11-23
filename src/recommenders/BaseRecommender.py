from abc import ABC, abstractmethod
from typing import List, Dict

class BaseRecommender(ABC):
    
    @abstractmethod
    def recommend(self, user_ids: List[int], k: int, history_filter: Dict[int, List[int]] = None) -> Dict[int, List[int]]:
        """
        Generate top-k recommendations for a list of users.

        Parameters:
        - user_ids (List[int]): List of user IDs to generate recommendations for.
        - k (int): Number of top recommendations to return for each user.
        - history_filter (Dict[int, List[int]], optional): A dictionary mapping user IDs to lists of item IDs that should be excluded from recommendations.

        Returns:
        - Dict[int, List[int]]: A dictionary mapping each user ID to a list of recommended item IDs.
        """
        pass
    