from tmdbv3api import TMDb, Movie
from movie_details import MovieDetails

class TmdbApiService:
    def __init__(self, tmdb_api_key: str):
        self.tmdb = TMDb()
        self.tmdb.api_key = tmdb_api_key
        self.movie = Movie()

    def fetch_movie_details(self, tmdb_movie_id: int) -> MovieDetails:
        movie_details = self.movie.details(tmdb_movie_id)

        return MovieDetails(
            title=movie_details.title,
            release_year=movie_details.release_date,
            genres=[genre['id'] for genre in movie_details.genres],
            runtime=movie_details.runtime,
            budget=movie_details.budget,
            revenue=movie_details.revenue,
            popularity=movie_details.popularity
        )