from typing import Literal
from datasets import Dataset, DatasetDict, load_dataset


class YambdaDataset:
  INTERACTIONS = frozenset(["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"])

  def __init__(
      self,
      dataset_type: Literal["flat", "sequential"] = "flat",
      dataset_size: Literal["50m", "500m", "5b"] = "50m"
  ):
    assert dataset_type in {"flat", "sequential"}
    assert dataset_size in {"50m", "500m", "5b"}
    self.dataset_type = dataset_type
    self.dataset_size = dataset_size

  def interaction(self, event_type: Literal["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"]) -> Dataset:
    assert event_type in self.INTERACTIONS
    return self._download(f"{self.dataset_type}/{self.dataset_size}", event_type)

  def audio_embeddings(self) -> Dataset:
    return self._download("", "embeddings")

  def album_item_mapping(self) -> Dataset:
    return self._download("", "album_item_mapping")

  def artist_item_mapping(self) -> Dataset:
    return self._download("", "artist_item_mapping")
  
  def get_random_items(self, event_type: Literal["likes", "listens", "multi_event", "dislikes", "unlikes", "undislikes"], n: int = 10) -> list[int]:
    interaction_df = self.interaction(event_type)
    items = interaction_df.to_pandas().sample(n)["item_id"].unique()
    return items.tolist()
    

  @staticmethod
  def _download(data_dir: str, file: str) -> Dataset:
    data = load_dataset("yandex/yambda", data_dir=data_dir, data_files=f"{file}.parquet")
    assert isinstance(data, DatasetDict)
    return data["train"]


dataset = YambdaDataset("flat", "50m")