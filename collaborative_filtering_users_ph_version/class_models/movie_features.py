import json
from typing import List, Tuple

from collaborative_filtering_users_ph_version.class_models.movie_features_type import MovieFeatureType

class MovieFeatures:
    def __init__(self, release_date: str, runtime: int, budget: int, revenue: int,
                 popularity: float, vote_average: float, vote_count: int, genres: List[int]):
        self.release_date: str = release_date
        self.runtime: int = runtime
        self.budget: int = budget
        self.revenue: int = revenue
        self.popularity: float = popularity
        self.vote_average: float = vote_average
        self.vote_count: int = vote_count
        self.genres: List[int] = genres

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def feature_types() -> List[Tuple[str, MovieFeatureType]]:
        return [
            ('release_date', MovieFeatureType.NUMERICAL),
            ('runtime', MovieFeatureType.NUMERICAL),
            ('budget', MovieFeatureType.NUMERICAL),
            ('revenue', MovieFeatureType.NUMERICAL),
            ('popularity', MovieFeatureType.NUMERICAL),
            ('vote_average', MovieFeatureType.NUMERICAL),
            ('vote_count', MovieFeatureType.NUMERICAL),
            ('genres', MovieFeatureType.CATEGORICAL)
        ]