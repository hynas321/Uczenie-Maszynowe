import json
from typing import List


class MovieDetails:
    def __init__(self, adult: bool, popularity: int, vote_average: int, vote_count: int, budget: int,
                 revenue: int, release_date: int, genres: List[int], runtime: int, spoken_languages: List[str],
                 production_companies: List[int], production_countries: List[str]):
        self.adult: bool = adult
        self.popularity: int = popularity
        self.vote_average: int = vote_average
        self.vote_count: int = vote_count
        self.budget: int = budget
        self.revenue: int = revenue
        self.runtime: int = runtime
        self.release_date: int = release_date
        self.genres: List[int] = genres
        self.spoken_languages: List[str] = spoken_languages
        self.production_companies: List[int] = production_companies
        self.production_countries: List[str] = production_countries

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)
