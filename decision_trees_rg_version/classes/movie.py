from decision_trees_rg_version.classes.movie_details import MovieDetails


class Movie:
    def __init__(self, movie_id: int, tmdb_movie_id: int, rating, movie_details: MovieDetails):
        self.movie_id: int = movie_id
        self.tmdb_movie_id: int = tmdb_movie_id
        self.rating: int = rating
        self.movie_details: MovieDetails = movie_details
