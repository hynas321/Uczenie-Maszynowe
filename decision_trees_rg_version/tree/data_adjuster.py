import numpy as np
from typing import List, Tuple
from decision_trees_rg_version.classes.movie import Movie


def preprocess_movie_details(movie_details) -> np.ndarray:
    """
    Preprocess the movie details and return them as a numpy array.
    """
    # Numerical features
    features = [
        int(movie_details.adult),
        movie_details.popularity,
        movie_details.vote_average,
        movie_details.vote_count,
        movie_details.budget,
        movie_details.revenue,
        movie_details.runtime,
    ]

    # Date feature
    from datetime import datetime
    release_date = datetime.strptime(movie_details.release_date, "%Y-%m-%d")
    days_since_release = (datetime.now() - release_date).days
    features.append(days_since_release)

    return np.array(features)  # Return as a numpy array


def prepare_data_for_testing(movies: List[Movie]) -> np.ndarray:
    """
    Prepare test data for predictions. Returns a 2D numpy array (features for each movie).
    """
    X = [preprocess_movie_details(movie.movie_details) for movie in movies]
    return np.array(X)  # Convert list of arrays into a 2D numpy array


def prepare_data_for_training(movies: List[Movie]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data. Returns:
    - X: 2D numpy array of features for each movie
    - y: 1D numpy array of target ratings
    """
    X, y = [], []
    for movie in movies:
        details = preprocess_movie_details(movie.movie_details)
        X.append(details)
        y.append(movie.rating)

    return np.array(X), np.array(y)  # Convert both features and labels into numpy arrays
