import pickle
import os
import numpy as np

from decision_trees_rg_version.classes.movie import Movie
from decision_trees_rg_version.classes.user import User
from decision_trees_rg_version.utils.csv_reader import load_csv_data
from decision_trees_rg_version.utils.data_sorter import get_movies_by_user
from decision_trees_rg_version.servies.tmdb_api import TMDBapi


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

        for rating in dict(task_data).get(user_id):
            movie_id, rate = rating
            movie_tmdb_id = movie_data.loc[movie_id, 'tmdb_movie_id']
            movie_details = tmdb_api_service.get_movie_details(movie_tmdb_id)

            movie = Movie(movie_id, movie_tmdb_id, None, movie_details)
            task_movies.append(movie)

        user = User(user_id, train_movies, task_movies)
        users.append(user)

        print(f"User with index = {index}")

    return users


def get_users():
    pickle_path = "users_data.pkl"

    data = fetch_or_generate_data(pickle_path, generate_data)

    return data
