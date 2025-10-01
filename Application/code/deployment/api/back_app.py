from typing import Union, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel
from Application.models.recommend import recommendation_system
from Application.data.yambda_dataset import dataset

class MusicItem(BaseModel):
    item_id: int
    
class ChosenMusicItems(BaseModel):
    items: list[MusicItem]

app = FastAPI(title="Yambda Recommendation Service", version="0.1.0")

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/recommend/{user_id}")
def recommend_music(
    user_id: int,
    chosen_items: Optional[ChosenMusicItems] = Body(None),
    top_n: int = 10,
):
    """Return top-N recommendations.

    If chosen_items is provided, treat those items as the user's known items.
    If omitted, the recommendation system will infer user history internally.
    """
    if chosen_items and chosen_items.items:
        user_items = {item.item_id for item in chosen_items.items}
    else:
        user_items = None  

    recommendations = recommendation_system.recommend_items_to_user(
        user_id=user_id,
        user_items=user_items,
        top_n=top_n,
    )
    to_return_recommendations = [
        {"item_id": item_id, "score": float(score)} for item_id, score in recommendations
    ]
    return {
        "user_id": user_id,
        "recommendations": to_return_recommendations,
        "count": len(to_return_recommendations),
        "used_input_items": bool(user_items),
    }
    
@app.get("/recommend/get_random_items")
def get_random_items():
    items = dataset.get_random_items("likes", n=10)
    return {
        "random_items": items,
        "count": len(items),
    }
