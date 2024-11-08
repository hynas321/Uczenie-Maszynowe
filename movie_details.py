import json

class MovieDetails:

    def __init__(self, title, release_year, genres, runtime, budget, revenue, popularity):
        self.title: str = title
        self.release_year: int = release_year
        self.genres: list[int] = genres
        self.runtime: int = runtime
        self.budget: int = budget
        self.revenue: int = revenue
        self.popularity: float = popularity

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)