import json
from typing import List


class MovieDetails:
    def __init__(self, adult: bool, popularity: int, vote_average: int, vote_count: int, budget: int,
                 revenue: int, release_date: int, genres: List[int], runtime: int, spoken_languages: List[str],
                 production_companies: List[int], production_countries: List[str]):
        self.adult = adult
        self.popularity = popularity
        self.vote_average = vote_average
        self.vote_count = vote_count
        self.budget = budget
        self.revenue = revenue
        self.runtime = runtime
        self.release_date = release_date
        self.genres = genres
        self.spoken_languages = spoken_languages
        self.production_companies = production_companies
        self.production_countries = production_countries

    def __repr__(self) -> str:
        return json.dumps(self.__dict__)
