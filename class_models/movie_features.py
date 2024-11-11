import json

from class_models.movie_feature_type import MovieFeatureType

class MovieFeatures:
    def __init__(self, release_date, runtime, budget, revenue, popularity, vote_average, genres):
        self.release_date = release_date
        self.runtime = runtime
        self.budget = budget
        self.revenue = revenue
        self.popularity = popularity
        self.vote_average = vote_average
        self.genres = genres

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def feature_types():
        return [
            ('release_date', MovieFeatureType.NUMERICAL),
            ('runtime', MovieFeatureType.NUMERICAL),
            ('budget', MovieFeatureType.NUMERICAL),
            ('revenue', MovieFeatureType.NUMERICAL),
            ('popularity', MovieFeatureType.NUMERICAL),
            ('vote_average', MovieFeatureType.NUMERICAL),
            ('genres', MovieFeatureType.CATEGORICAL)
        ]