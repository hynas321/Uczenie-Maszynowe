import pandas as pd
from collections import defaultdict
from typing import List, Tuple


def get_movies_by_user(data: pd.DataFrame) -> List[Tuple[int, List[Tuple[int, int]]]]:
    user_ratings = defaultdict(list)

    for _, row in data.iterrows():
        user_id = row["user_id"]
        movie_id = row["movie_id"]
        rating = row["rating"]

        user_ratings[user_id].append((movie_id, rating))

    user_ratings_list = [(user_id, ratings) for user_id, ratings in user_ratings.items()]

    return user_ratings_list

