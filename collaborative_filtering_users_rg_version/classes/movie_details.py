import json


class MovieDetails:
    def __init__(self, adult: int, popularity: int, vote_average: int, vote_count: int, budget: int,
                 revenue: int, release_date: int, runtime: int, ):
        self.adult: int = adult
        self.popularity: int = popularity
        self.vote_average: int = vote_average
        self.vote_count: int = vote_count
        self.budget: int = budget
        self.revenue: int = revenue
        self.runtime: int = runtime
        self.release_date: int = release_date

    def __str__(self):
        return f'Popularity: {self.popularity}, Budget: {self.budget}, Release_date: {self.release_date}'

    def __repr__(self):
        return f'Popularity: {self.popularity}, Budget: {self.budget}, Release_date: {self.release_date}'