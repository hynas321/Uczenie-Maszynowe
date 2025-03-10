import pandas as pd
from collections import defaultdict, Counter
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


def divide_list(lst, slices_number=5):
    sublist_size = len(lst) // slices_number
    remainder = len(lst) % slices_number

    lists = []
    start_index = 0

    for i in range(slices_number):
        end_index = start_index + sublist_size + (1 if i < remainder else 0)
        sublist = lst[start_index:end_index]
        lists.append(sublist)
        start_index = end_index

    return lists


def get_k_best_point(points, k: int):
    sorted_points = sorted(points, key=lambda x: x[0])

    k_points = sorted_points[:k]

    ratings = [point[1] for point in k_points]

    frequency = Counter(ratings)

    most_common = frequency.most_common()
    max_count = most_common[0][1]

    most_frequent = [rating for rating, count in most_common if count == max_count]

    return most_frequent[0]
