from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd

from similarity_users_rg_version.classes.movie import Movie
from similarity_users_rg_version.classes.user import User


def generate_users(train_data: pd.DataFrame, task_data: pd.DataFrame):
    task_data: list[tuple[int, list[tuple[int, int]]]] = get_movies_by_user(task_data)
    train_data: list[tuple[int, list[tuple[int, int]]]] = get_movies_by_user(train_data)

    users = []

    for index, user_data in enumerate(train_data, start=1):

        train_movies = []
        task_movies = []

        user_id, ratings = user_data
        ratings_array = np.array(ratings)

        for rating in ratings_array:
            movie_id, rate = rating

            movie = Movie(movie_id, rate)
            train_movies.append(movie)

        for rating in dict(task_data).get(user_id):
            movie_id, rate = rating

            movie = Movie(movie_id, None)
            task_movies.append(movie)

        user = User(user_id, train_movies, task_movies)
        users.append(user)

    return users


def get_movies_by_user(data: pd.DataFrame) -> list[Tuple[int, list[Tuple[int, int]]]]:
    user_ratings = defaultdict(list)

    for _, row in data.iterrows():
        user_id = int(row["user_id"])
        movie_id = int(row["movie_id"])
        if pd.isna(row["rating"]):
            rating = None
        else:
            rating = int(row["rating"])

        user_ratings[user_id].append((movie_id, rating))

    user_ratings_list = [(user_id, ratings) for user_id, ratings in user_ratings.items()]

    return user_ratings_list
