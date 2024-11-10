import json

class MovieFeatures:
    def __init__(self, release_date, genres, runtime, budget, revenue, popularity):
        self.release_date = release_date
        self.genres = genres
        self.runtime = runtime
        self.budget = budget
        self.revenue = revenue
        self.popularity = popularity

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def feature_types():
        return [
            ('release_date', 'numerical'),
            ('runtime', 'numerical'),
            ('budget', 'numerical'),
            ('revenue', 'numerical'),
            ('popularity', 'numerical'),
            ('genres', 'categorical')
        ]