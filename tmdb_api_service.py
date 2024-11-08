from tmdbv3api import TMDb, Movie

class TmdbApiService:
    def __init__(self, tmdb_api_key: str):
        self.tmdb = TMDb()
        self.tmdb.api_key = tmdb_api_key
        self.movie = Movie()

    def fetch_movie_details(self, tmdb_movie_id: int):
        return self.movie.details(tmdb_movie_id)