import json

from collaborative_filtering_users_rg_version.classes.movie_details import MovieDetails


class Movie:
    def __init__(self, movie_id: int, tmdb_movie_id: int, rating, movie_details: MovieDetails):
        self.movie_id: int = movie_id
        self.tmdb_movie_id: int = tmdb_movie_id
        self.rating: int = rating
        self.movie_details: MovieDetails = movie_details

    def __str__(self) -> str:
        return f'Movie with id: {self.movie_id}, rating: {self.rating}, details: {self.movie_details}'

    def __repr__(self) -> str:
        return f'Movie with id: {self.movie_id}, rating: {self.rating}, details: {self.movie_details}'

    def get_features(self):
        return [self.movie_details.adult, self.movie_details.popularity, self.movie_details.vote_average,
                self.movie_details.vote_count, self.movie_details.budget, self.movie_details.release_date,
                self.movie_details.runtime, self.movie_details.release_date]
