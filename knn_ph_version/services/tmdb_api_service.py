from tmdbv3api import TMDb, Movie

from knn_ph_version.class_models.movie_features import MovieFeatures


class TmdbApiService:
    def __init__(self, tmdb_api_key: str) -> None:
        self.tmdb = TMDb()
        self.tmdb.api_key = tmdb_api_key
        self.movie = Movie()

    def fetch_movie_details(self, tmdb_movie_id: int) -> MovieFeatures:
        movie_details = self.movie.details(tmdb_movie_id)

        return MovieFeatures(
            release_date=movie_details.release_date,
            runtime=movie_details.runtime,
            budget=movie_details.budget,
            revenue=movie_details.revenue,
            popularity=movie_details.popularity,
            vote_average=movie_details.vote_average,
            vote_count=movie_details.vote_count,
            genres=[genre['id'] for genre in movie_details.genres],
        )