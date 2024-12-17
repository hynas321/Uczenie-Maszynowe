from datetime import datetime

from tmdbv3api import TMDb, Movie

from collaborative_filtering_users_rg_version.classes.movie_details import MovieDetails


def calculate_days_since_release(release_date_str):

    date_format = "%Y-%m-%d"
    release_date = datetime.strptime(release_date_str, date_format)
    today = datetime.now()
    days_since_release = (today - release_date).days
    return days_since_release


class TMDBapi:
    def __init__(self):
        self.tmdb = TMDb()
        self.tmdb.api_key = '0a35be2e46bd817c47c0cf38f1c8dfdc'
        self.movie = Movie()

    def get_movie_details(self, tmdb_movie_id: int):
        movie_details = self.movie.details(tmdb_movie_id)

        return MovieDetails(
            adult=int(movie_details.adult),
            popularity=movie_details.popularity,
            vote_average=movie_details.vote_average,
            vote_count=movie_details.vote_count,
            budget=movie_details.budget,
            revenue=movie_details.revenue,
            release_date=calculate_days_since_release(movie_details.release_date),
            runtime=movie_details.runtime,
        )
