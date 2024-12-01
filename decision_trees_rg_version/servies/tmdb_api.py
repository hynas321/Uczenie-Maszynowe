from tmdbv3api import TMDb, Movie

from knn_rg_version.classes.movie_details import MovieDetails


class TMDBapi:
    def __init__(self):
        self.tmdb = TMDb()
        self.tmdb.api_key = '0a35be2e46bd817c47c0cf38f1c8dfdc'
        self.movie = Movie()

    def get_movie_details(self, tmdb_movie_id: int):
        movie_details = self.movie.details(tmdb_movie_id)

        return MovieDetails(
            adult=movie_details.adult,
            popularity=movie_details.popularity,
            vote_average=movie_details.vote_average,
            vote_count=movie_details.vote_count,
            budget=movie_details.budget,
            revenue=movie_details.revenue,
            release_date=movie_details.release_date,
            genres=[genre['id'] for genre in movie_details.genres],
            runtime=movie_details.runtime,
            spoken_languages=[spoken_language['iso_639_1'] for spoken_language in
                              movie_details.spoken_languages],
            production_companies=[production_company['id'] for production_company in
                                  movie_details.production_companies],
            production_countries=[production_country['iso_3166_1'] for production_country in
                                  movie_details.production_countries],
        )
