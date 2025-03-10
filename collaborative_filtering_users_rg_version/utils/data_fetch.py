import pickle
import os
import numpy as np

from collaborative_filtering_users_rg_version.classes.movie import Movie
from collaborative_filtering_users_rg_version.classes.user import User
from collaborative_filtering_users_rg_version.service.tmdb_api import TMDBapi
from collaborative_filtering_users_rg_version.utils.csv_reader import load_csv_data
from collaborative_filtering_users_rg_version.utils.data_sorter import get_movies_by_user


def fetch_or_generate_data(pickle_file_path, generate_data_function):
    if os.path.exists(pickle_file_path):
        print(f"Loading data from {pickle_file_path}...")
        with open(pickle_file_path, "rb") as file:
            data = pickle.load(file)
    else:
        print(f"{pickle_file_path} not found. Generating new data...")
        data = generate_data_function()
        with open(pickle_file_path, "wb") as file:
            pickle.dump(data, file)
        print(f"Data saved to {pickle_file_path}.")
    return data


def generate_data():
    movie_data, task_data_, train_data_ = load_csv_data()
    task_data: list[tuple[int, list[tuple[int, int]]]] = get_movies_by_user(task_data_)
    train_data: list[tuple[int, list[tuple[int, int]]]] = get_movies_by_user(train_data_)

    tmdb_api_service = TMDBapi()
    users = []

    for index, user_data in enumerate(train_data, start=1):

        train_movies = []
        task_movies = []

        user_id, ratings = user_data
        ratings_array = np.array(ratings)

        for rating in ratings_array:
            movie_id, rate = rating
            movie_tmdb_id = movie_data.loc[movie_id, 'tmdb_movie_id']
            movie_details = tmdb_api_service.get_movie_details(movie_tmdb_id)

            movie = Movie(movie_id, movie_tmdb_id, rate, movie_details)
            train_movies.append(movie)

        standardize_features(train_movies)
        print(train_movies)

        for rating in dict(task_data).get(user_id):
            movie_id, rate = rating
            movie_tmdb_id = movie_data.loc[movie_id, 'tmdb_movie_id']
            movie_details = tmdb_api_service.get_movie_details(movie_tmdb_id)

            movie = Movie(movie_id, movie_tmdb_id, None, movie_details)
            task_movies.append(movie)

        standardize_features(task_movies)

        user = User(user_id, train_movies, task_movies)
        users.append(user)

        print(f"User with index = {index}")

    return users


import numpy as np


def standardize_features(movies):
    feature_matrix = np.array([
        [
            movie.movie_details.adult,
            movie.movie_details.popularity,
            movie.movie_details.vote_average,
            movie.movie_details.vote_count,
            movie.movie_details.budget,
            movie.movie_details.revenue,
            movie.movie_details.release_date,
            movie.movie_details.runtime,
        ]
        for movie in movies
    ])

    means = feature_matrix.mean(axis=0)
    stds = feature_matrix.std(axis=0)

    stds[stds == 0] = 1

    standardized_matrix = (feature_matrix - means) / stds

    for i, movie in enumerate(movies):
        movie.movie_details.adult = standardized_matrix[i, 0]
        movie.movie_details.popularity = standardized_matrix[i, 1]
        movie.movie_details.vote_average = standardized_matrix[i, 2]
        movie.movie_details.vote_count = standardized_matrix[i, 3]
        movie.movie_details.budget = standardized_matrix[i, 4]
        movie.movie_details.revenue = standardized_matrix[i, 5]
        movie.movie_details.release_date = standardized_matrix[i, 6]
        movie.movie_details.runtime = standardized_matrix[i, 7]

    return movies


def get_users():
    pickle_path = "users_data.pkl"

    data = fetch_or_generate_data(pickle_path, generate_data)

    return data
