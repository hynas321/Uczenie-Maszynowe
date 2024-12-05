from typing import Dict, Tuple
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def compute_user_similarity(user_ratings: pd.Series, other_ratings: pd.Series) -> float:
    common_movies = user_ratings.notna() & other_ratings.notna()

    if common_movies.sum() > 0:
        differences = (user_ratings[common_movies] - other_ratings[common_movies]).abs()
        max_diff = differences.max()
        min_diff = differences.min()
        variance_penalty = differences.std()

        similarity_score = 1 / (1 + max_diff + min_diff + variance_penalty)
        return similarity_score

    return 0


def compute_similarity_matrix(user_movie_matrix: DataFrame) -> Dict[int, Dict[int, float]]:
    similarity_matrix = {}
    for user_id in tqdm(user_movie_matrix.index, desc="Computing Similarity Matrix"):
        similarity_matrix[user_id] = {}
        user_ratings = user_movie_matrix.loc[user_id]

        for other_user in user_movie_matrix.index:
            if user_id != other_user:
                other_ratings = user_movie_matrix.loc[other_user]
                similarity_score = compute_user_similarity(user_ratings, other_ratings)
                similarity_matrix[user_id][other_user] = similarity_score

    return similarity_matrix


def predict_user_movie_rating(
    user_id: int, movie_id: int, similarity_matrix: Dict[int, Dict[int, float]],
    user_movie_matrix: DataFrame
) -> float:
    user_mode = user_movie_matrix.loc[user_id].mode().iloc[0] if not user_movie_matrix.loc[user_id].mode().empty else 3

    if movie_id not in user_movie_matrix.columns:
        return user_mode

    similar_users = similarity_matrix[user_id]
    movie_ratings = user_movie_matrix[movie_id]

    neighbors = {
        neighbor: sim for neighbor, sim in similar_users.items() if not pd.isna(movie_ratings.get(neighbor, None))
    }

    if not neighbors:
        return user_mode

    numerator = sum(sim * (movie_ratings[neighbor] - (
        user_movie_matrix.loc[neighbor].mode().iloc[0]
        if not user_movie_matrix.loc[neighbor].mode().empty else 3))
                    for neighbor, sim in neighbors.items())
    denominator = sum(abs(sim) for sim in neighbors.values())

    predicted_rating = user_mode + (numerator / denominator if denominator > 0 else 0)
    return max(0, min(5, predicted_rating))


def validate_user_predictions(
    user_id: int, train_data: DataFrame, similarity_matrix: Dict[int, Dict[int, float]],
    user_movie_matrix: DataFrame
) -> float:
    user_ratings = train_data[train_data['user_id'] == user_id]

    correct, total = 0, 0
    for _, row in user_ratings.iterrows():
        movie_id, actual_rating = row['movie_id'], row['rating']
        predicted_rating = predict_user_movie_rating(user_id, movie_id, similarity_matrix, user_movie_matrix)

        if round(predicted_rating) == actual_rating:
            correct += 1
        total += 1

    return correct / total if total > 0 else None


def fill_ratings(
    task_data: DataFrame, similarity_matrix: Dict[int, Dict[int, float]], user_movie_matrix: DataFrame
) -> DataFrame:
    task_data_filled = task_data.copy()

    for idx, row in tqdm(task_data.iterrows(), desc="Filling Ratings", total=len(task_data)):
        if pd.isna(row['rating']):
            user_id, movie_id = row['user_id'], row['movie_id']
            predicted_rating = predict_user_movie_rating(user_id, movie_id, similarity_matrix, user_movie_matrix)
            task_data_filled.at[idx, 'rating'] = predicted_rating

    return task_data_filled


def validate_and_process_ratings(
    train_data: DataFrame, task_data: DataFrame
) -> Tuple[DataFrame, Dict[int, float]]:
    user_movie_matrix = train_data.pivot(index='user_id', columns='movie_id', values='rating')
    similarity_matrix = compute_similarity_matrix(user_movie_matrix)

    user_accuracies = {}
    for user_id in tqdm(user_movie_matrix.index, desc="Validating Predictions"):
        accuracy = validate_user_predictions(user_id, train_data, similarity_matrix, user_movie_matrix)
        if accuracy is not None:
            user_accuracies[user_id] = accuracy

    task_data_filled = fill_ratings(task_data, similarity_matrix, user_movie_matrix)

    return task_data_filled, user_accuracies